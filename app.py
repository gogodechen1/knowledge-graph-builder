from flask import Flask, jsonify, request, render_template, Response, stream_with_context
from flask_cors import CORS
import json
import networkx as nx
import requests
import os
from dotenv import load_dotenv
import time
import threading
from queue import Queue
import uuid
from werkzeug.utils import secure_filename

app = Flask(__name__)
CORS(app)  # 启用CORS
G = nx.Graph()

# 配置文件上传
UPLOAD_FOLDER = 'static/uploads'
ALLOWED_EXTENSIONS = {'png'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max-limit

# 加载环境变量
load_dotenv()
XAI_API_KEY = "xai-6SgXFpFCNZSLjgAGPDcdD0ggHToTsAtZjWJhemrW7QoHSOty9LpVK9S3BMtsJIFmMKew13ITydmr50Tt"
XAI_API_URL = "https://api.x.ai/v1/chat/completions"

# 用于存储正在处理的节点请求
processing_nodes = set()
# 用于存储每个请求的状态
request_status = {}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def load_initial_knowledge():
    """加载初始知识图谱"""
    try:
        with open('initial_knowledge.json', 'r', encoding='utf-8') as f:
            data = json.load(f)
            for node in data['nodes']:
                G.add_node(node['label'])
            for edge in data['edges']:
                G.add_edge(edge['from'], edge['to'], relationship=edge.get('label', ''))
    except Exception as e:
        print(f"Error loading initial knowledge: {e}")

def stream_ai_analysis(node_name, node_description, request_id):
    """流式AI分析过程"""
    headers = {
        "Authorization": f"Bearer {XAI_API_KEY}",
        "Content-Type": "application/json"
    }
    
    # 构建当前图谱的节点和关系信息
    current_nodes = list(G.nodes())
    current_edges = [(u, v, G[u][v].get('relationship', '')) for u, v in G.edges()]
    
    prompt = f"""作为一个知识图谱分析专家，请帮我分析节点"{node_name}"（描述：{node_description}）。
    
    当前图谱信息：
    节点列表：{current_nodes}
    关系列表：{current_edges}
    
    请按以下步骤进行分析并输出：
    1. 理解概念：分析这个概念的核心含义
    2. 寻找关系：思考与现有节点可能的关联
    3. 扩展联想：提出新的相关概念
    4. 形成结论：给出最终的关系建议
    
    请确保每一步都输出以下JSON格式：
    {{"process": "当前步骤的分析内容", "type": "思考/结论", "relationships": []}}
    
    在最后的结论中，relationships应该是一个包含关系的列表，每个关系格式为：
    {{"source": "源节点", "target": "目标节点", "relationship": "关系描述"}}
    """
    
    try:
        response = requests.post(
            XAI_API_URL,
            headers=headers,
            json={
                "model": "grok-beta",
                "messages": [{"role": "user", "content": prompt}],
                "stream": True
            },
            stream=True
        )
        
        if response.status_code != 200:
            error_msg = {"error": f"AI服务返回错误: {response.status_code}", "type": "error"}
            yield f"data: {json.dumps(error_msg)}\n\n"
            return
            
        current_chunk = ""
        for line in response.iter_lines():
            if line:
                if line.startswith(b"data: "):
                    json_str = line[6:].decode('utf-8')
                    if json_str.strip() == "[DONE]":
                        continue
                    try:
                        chunk_data = json.loads(json_str)
                        if "choices" in chunk_data and chunk_data["choices"]:
                            content = chunk_data["choices"][0].get("delta", {}).get("content", "")
                            if content:
                                current_chunk += content
                                if "}" in current_chunk:
                                    try:
                                        # 尝试解析完整的JSON对象
                                        result = json.loads(current_chunk)
                                        # 更新请求状态
                                        request_status[request_id] = result
                                        # 生成SSE事件
                                        yield f"data: {json.dumps(result)}\n\n"
                                        current_chunk = ""
                                    except json.JSONDecodeError:
                                        continue
                    except json.JSONDecodeError:
                        continue
        
        # 处理完成后，发送完成信号
        yield f"data: {json.dumps({'type': 'complete'})}\n\n"
        
    except Exception as e:
        error_msg = {"error": str(e), "type": "error"}
        yield f"data: {json.dumps(error_msg)}\n\n"
    finally:
        # 清理处理状态
        if node_name in processing_nodes:
            processing_nodes.remove(node_name)
        if request_id in request_status:
            del request_status[request_id]

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/get_graph')
def get_graph():
    try:
        nodes = [{"id": node, "label": node} for node in G.nodes()]
        edges = [{"from": u, "to": v, "label": G[u][v].get('relationship', '')} 
                for u, v in G.edges()]
        
        return jsonify({
            "nodes": nodes,
            "edges": edges
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/ai/status', methods=['GET'])
def check_ai_status():
    try:
        headers = {
            "Authorization": f"Bearer {XAI_API_KEY}",
            "Content-Type": "application/json"
        }
        response = requests.post(
            XAI_API_URL,
            headers=headers,
            json={
                "model": "grok-beta",
                "messages": [{"role": "user", "content": "test"}],
                "max_tokens": 1
            }
        )
        return jsonify({"status": "online" if response.status_code == 200 else "offline"})
    except Exception as e:
        print(f"AI status check error: {e}")
        return jsonify({"status": "offline"})

@app.route('/add_node', methods=['POST'])
def add_node():
    try:
        data = request.json
        if not data:
            return jsonify({"error": "No data provided"}), 400
            
        node_name = data.get('name')
        node_description = data.get('description', '')
        
        if not node_name:
            return jsonify({"error": "Node name is required"}), 400
        
        # 检查节点是否已经在处理中
        if node_name in processing_nodes:
            return jsonify({"error": "Node is already being processed"}), 409
        
        # 生成请求ID
        request_id = str(uuid.uuid4())
        processing_nodes.add(node_name)
        
        # 返回流式响应
        return Response(
            stream_with_context(stream_ai_analysis(node_name, node_description, request_id)),
            content_type='text/event-stream',
            headers={
                'Cache-Control': 'no-cache',
                'Connection': 'keep-alive',
                'X-Accel-Buffering': 'no'
            }
        )
    except Exception as e:
        if node_name in processing_nodes:
            processing_nodes.remove(node_name)
        return jsonify({"error": str(e)}), 500

@app.route('/confirm_node', methods=['POST'])
def confirm_node():
    """确认添加节点和关系"""
    try:
        data = request.json
        if not data:
            return jsonify({"error": "No data provided"}), 400
            
        node_name = data.get('name')
        relationships = data.get('relationships', [])
        
        if not node_name:
            return jsonify({"error": "Node name is required"}), 400
        
        # 添加新节点
        if node_name not in G.nodes:
            G.add_node(node_name)
        
        # 添加关系
        for rel in relationships:
            source = rel.get('source')
            target = rel.get('target')
            relationship = rel.get('relationship')
            
            if not all([source, target, relationship]):
                continue
                
            # 确保源节点和目标节点都存在
            if source not in G.nodes:
                G.add_node(source)
            if target not in G.nodes:
                G.add_node(target)
                
            # 添加边和关系属性
            G.add_edge(source, target, relationship=relationship)
        
        # 返回更新后的图数据
        nodes = [{"id": node, "label": node} for node in G.nodes()]
        edges = [{"from": u, "to": v, "label": G[u][v].get('relationship', '')} 
                for u, v in G.edges()]
        
        return jsonify({
            "success": True,
            "graph": {
                "nodes": nodes,
                "edges": edges
            }
        })
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        # 确保上传目录存在
        os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        return jsonify({
            'success': True,
            'filename': filename,
            'url': f'/static/uploads/{filename}'
        })
    return jsonify({'error': 'File type not allowed'}), 400

# 初始化知识图谱
load_initial_knowledge()

if __name__ == '__main__':
    app.run(debug=True)
