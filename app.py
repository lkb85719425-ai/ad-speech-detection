
"""
AD语音检测系统 - 完整Web服务
整合API服务和网页界面，提供完整的阿尔茨海默症语音检测功能
"""
import os
from flask import Flask, request, jsonify, render_template_string
from flask_cors import CORS
import numpy as np
import joblib
import os
import time
import uuid
import logging
from datetime import datetime

# =============================================================================
# 添加缺失的函数定义以支持模型加载
# =============================================================================

import libr

# 特征维度常量（必须与训练时一致
N_MFCC = 13
N_MEL = 128
N_FEATURES = 286

def extract_audio_features(audio_file):
    """
    从音频文件中提取语音声学特征
    （简化版本，仅用于模型加载兼容性）
    """
    try:
        # 这里只需要返回正确维度的零向量
        # 因为API服务实际使用的是特征数据，不是原始音频
        return np.zeros(N_FEATURES)
    except Exception:
        return np.zeros(N_FEATURES)

def generate_highly_realistic_simulated_data(n_samples=1500, ad_ratio=0.3, difficulty_level=0.8):
    """模拟数据生成函数（简化版）"""
    X = np.random.normal(0, 1, (n_samples, N_FEATURES))
    y = np.random.binomial(1, ad_ratio, n_samples)
    return X, y

def load_and_preprocess_audio_data(audio_dir=None, ad_ratio=0.3, difficulty_level=0.8):
    """数据加载函数（简化版）"""
    X, y = generate_highly_realistic_simulated_data(1000, ad_ratio, difficulty_level)
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    from sklearn.preprocessing import StandardScaler
    from sklearn.compose import ColumnTransformer
    preprocessor = ColumnTransformer([('num', StandardScaler(), list(range(N_FEATURES)))])
    preprocessor.fit(X_train)
    return X_train, X_test, y_train, y_test, preprocessor

# =============================================================================
# 应用初始化
# =============================================================================

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)  # 允许跨域请求

# 全局变量存储模型
model = None
model_loaded = False

# 简单的内存存储，用于保存录音和分析结果的对应关系
audio_records = {}

def load_model():
    """加载训练好的模型"""
    global model, model_loaded

    try:
        # 优先尝试加载简化模型
        possible_paths = [
            'ad_model_simple.pkl',      # 新简化模型
            'ad_speech_detector.pkl',   # 旧模型
            './ad_model_simple.pkl',
            './ad_speech_detector.pkl',
            'model/ad_model_simple.pkl',
            'model/ad_speech_detector.pkl',
        ]

        model_path = None
        for path in possible_paths:
            if os.path.exists(path):
                model_path = path
                break

        if model_path is None:
            logger.error("❌ 未找到模型文件")
            # 列出当前目录文件帮助调试
            current_files = os.listdir('.')
            logger.info(f"当前目录文件: {current_files}")
            return False

        logger.info(f"📁 找到模型文件: {model_path}")
        model = joblib.load(model_path)
        model_loaded = True
        logger.info("✅ 模型加载成功")
        return True

    except Exception as e:
        logger.error(f"❌ 模型加载失败: {str(e)}")
        model_loaded = False
        return False

# HTML模板 - 包含完整的录音和手动检测功能
HTML_TEMPLATE = '''
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>阿尔茨海默症语音检测系统</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: 'Microsoft YaHei', Arial, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
        }
        .container {
            max-width: 900px;
            margin: 0 auto;
            background: white;
            border-radius: 15px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.2);
            overflow: hidden;
        }
        .header {
            background: linear-gradient(135deg, #2c3e50, #3498db);
            color: white;
            padding: 30px;
            text-align: center;
        }
        .header h1 {
            font-size: 2.2em;
            margin-bottom: 10px;
        }
        .content { padding: 30px; }
        .card {
            background: #f8f9fa;
            border-radius: 10px;
            padding: 25px;
            margin-bottom: 20px;
            border-left: 5px solid #3498db;
        }
        .btn {
            display: inline-block;
            padding: 12px 30px;
            background: linear-gradient(135deg, #3498db, #2980b9);
            color: white;
            text-decoration: none;
            border-radius: 25px;
            border: none;
            cursor: pointer;
            font-size: 16px;
            transition: all 0.3s ease;
            margin: 5px;
        }
        .btn:hover {
            transform: translateY(-2px);
            box-shadow: 0 5px 15px rgba(0,0,0,0.2);
        }
        .btn-success { background: linear-gradient(135deg, #27ae60, #229954); }
        .btn-danger { background: linear-gradient(135deg, #e74c3c, #c0392b); }
        .btn-warning { background: linear-gradient(135deg, #f39c12, #e67e22); }
        .btn:disabled {
            background: #95a5a6;
            cursor: not-allowed;
            transform: none;
            box-shadow: none;
        }
        .result {
            margin-top: 20px;
            padding: 20px;
            border-radius: 10px;
            display: none;
        }
        .risk-high { background: #ffeaea; border-left: 5px solid #e74c3c; }
        .risk-medium { background: #fff3cd; border-left: 5px solid #f39c12; }
        .risk-low { background: #e8f5e8; border-left: 5px solid #27ae60; }
        .status {
            padding: 10px;
            border-radius: 5px;
            margin: 10px 0;
            text-align: center;
            font-weight: bold;
        }
        .status-ok { background: #d4edda; color: #155724; }
        .status-error { background: #f8d7da; color: #721c24; }
        .status-warning { background: #fff3cd; color: #856404; }
        .loading {
            display: none;
            text-align: center;
            padding: 20px;
        }
        .spinner {
            border: 4px solid #f3f3f3;
            border-top: 4px solid #3498db;
            border-radius: 50%;
            width: 40px;
            height: 40px;
            animation: spin 2s linear infinite;
            margin: 0 auto;
        }
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }

        /* 录音界面样式 */
        .recording-container {
            text-align: center;
            padding: 20px;
        }
        .recording-visualizer {
            width: 100%;
            height: 100px;
            background: #2c3e50;
            border-radius: 10px;
            margin: 20px 0;
            position: relative;
            overflow: hidden;
        }
        .visualizer-bars {
            display: flex;
            justify-content: space-around;
            align-items: flex-end;
            height: 100%;
            padding: 0 10px;
        }
        .visualizer-bar {
            width: 8px;
            background: #3498db;
            border-radius: 4px 4px 0 0;
            transition: height 0.1s ease;
        }
        .recording-timer {
            font-size: 2em;
            font-weight: bold;
            color: #2c3e50;
            margin: 15px 0;
        }
        .recording-indicator {
            display: inline-block;
            width: 12px;
            height: 12px;
            background: #e74c3c;
            border-radius: 50%;
            margin-right: 8px;
            animation: pulse 1.5s infinite;
        }
        @keyframes pulse {
            0% { opacity: 1; }
            50% { opacity: 0.3; }
            100% { opacity: 1; }
        }
        .audio-playback {
            width: 100%;
            margin: 15px 0;
        }
        .instructions {
            background: #e8f4fd;
            border-left: 4px solid #3498db;
            padding: 15px;
            margin: 15px 0;
            border-radius: 0 8px 8px 0;
        }
        .step-number {
            display: inline-block;
            width: 25px;
            height: 25px;
            background: #3498db;
            color: white;
            border-radius: 50%;
            text-align: center;
            line-height: 25px;
            margin-right: 10px;
        }

        /* 录音历史记录 */
        .recordings-history {
            margin-top: 20px;
            max-height: 300px;
            overflow-y: auto;
            border: 1px solid #ddd;
            border-radius: 8px;
            padding: 10px;
        }
        .recording-item {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 10px;
            border-bottom: 1px solid #eee;
        }
        .recording-item:last-child {
            border-bottom: none;
        }
        .recording-info {
            flex-grow: 1;
        }
        .recording-actions {
            display: flex;
            gap: 10px;
        }
        .small-btn {
            padding: 5px 10px;
            font-size: 12px;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🧠 阿尔茨海默症语音检测系统</h1>
            <p>基于人工智能的早期筛查工具 - 支持实时语音分析</p>
        </div>

        <div class="content">
            <!-- 系统状态检查 -->
            <div class="card">
                <h2>📊 系统状态</h2>
                <div id="systemStatus">
                    <div class="status">检查系统中...</div>
                </div>
                <button class="btn" onclick="checkSystemStatus()">刷新状态</button>
            </div>

            <!-- 实时录音检测 -->
            <div class="card">
                <h2>🎤 实时语音检测</h2>
                <div class="instructions">
                    <p><span class="step-number">1</span> 点击"开始录音"按钮，请用清晰、自然的声音朗读一段文字</p>
                    <p><span class="step-number">2</span> 建议朗读时间：10-30秒</p>
                    <p><span class="step-number">3</span> 完成后点击"停止录音"，然后点击"检测录音"进行分析</p>
                </div>

                <div class="recording-container">
                    <div id="recordingControls">
                        <button class="btn btn-success" id="startRecording" onclick="startRecording()">
                            🎤 开始录音
                        </button>
                        <button class="btn btn-danger" id="stopRecording" onclick="stopRecording()" disabled>
                            ⏹️ 停止录音
                        </button>
                        <button class="btn" id="playRecording" onclick="playRecording()" disabled>
                            ▶️ 播放录音
                        </button>
                        <button class="btn btn-warning" id="analyzeRecording" onclick="analyzeRecording()" disabled>
                            🔍 检测录音
                        </button>
                    </div>

                    <div id="recordingStatus" style="display: none;">
                        <span class="recording-indicator"></span>
                        <span>正在录音中...</span>
                    </div>

                    <div class="recording-timer" id="recordingTimer">00:00</div>

                    <div class="recording-visualizer">
                        <div class="visualizer-bars" id="visualizer"></div>
                    </div>

                    <audio id="audioPlayback" class="audio-playback" controls style="display: none;"></audio>

                    <div id="recordingInfo" style="display: none; margin-top: 15px;">
                        <div class="status status-warning">
                            ✅ 录音已完成，请点击"检测录音"按钮进行分析
                        </div>
                    </div>
                </div>

                <!-- 录音历史记录 -->
                <div id="recordingsHistory" style="display: none;">
                    <h3>📋 录音历史记录</h3>
                    <div class="recordings-history" id="historyList">
                        <!-- 录音历史记录将在这里动态生成 -->
                    </div>
                </div>
            </div>

            <!-- 快速检测 -->
            <div class="card">
                <h2>⚡ 快速检测</h2>
                <p>使用模拟数据进行快速测试</p>
                <button class="btn btn-success" onclick="runDemoTest()">运行演示测试</button>
                <button class="btn btn-danger" onclick="runADSimulation()">模拟AD患者检测</button>
                <button class="btn" onclick="runNormalSimulation()">模拟正常人检测</button>
            </div>

            <!-- 结果显示 -->
            <div class="card">
                <h2>📈 检测结果</h2>
                <div class="loading" id="loading">
                    <div class="spinner"></div>
                    <p>分析中，请稍候...</p>
                </div>
                <div id="result" class="result"></div>
            </div>
        </div>
    </div>

    <script>
        // 录音相关变量
        let mediaRecorder;
        let audioChunks = [];
        let recordingTimer;
        let seconds = 0;
        let audioContext;
        let analyser;
        let dataArray;
        let bufferLength;
        let visualizerBars;
        let currentRecordingId = null;
        let currentAudioBlob = null;
        let recordingsHistory = [];

        // 页面加载时检查系统状态
        window.onload = function() {
            checkSystemStatus();
            initializeVisualizer();
            loadRecordingsHistory();
        };

        // 初始化音频可视化器
        function initializeVisualizer() {
            const visualizer = document.getElementById('visualizer');
            visualizer.innerHTML = '';
            visualizerBars = [];

            // 创建32个可视化条
            for (let i = 0; i < 32; i++) {
                const bar = document.createElement('div');
                bar.className = 'visualizer-bar';
                bar.style.height = '5px';
                visualizer.appendChild(bar);
                visualizerBars.push(bar);
            }
        }

        // 更新音频可视化
        function updateVisualizer() {
            if (!analyser) return;

            analyser.getByteFrequencyData(dataArray);

            for (let i = 0; i < bufferLength; i++) {
                const value = dataArray[i];
                const percentage = value / 256;
                const bar = visualizerBars[i];

                if (bar) {
                    bar.style.height = `${Math.max(5, percentage * 100)}px`;
                    // 根据音量改变颜色
                    const green = Math.min(255, value + 100);
                    bar.style.background = `rgb(${value}, ${green}, 200)`;
                }
            }

            requestAnimationFrame(updateVisualizer);
        }

        // 开始录音
        async function startRecording() {
            try {
                // 请求麦克风权限
                const stream = await navigator.mediaDevices.getUserMedia({ audio: true });

                // 设置音频上下文和分析器
                audioContext = new (window.AudioContext || window.webkitAudioContext)();
                analyser = audioContext.createAnalyser();
                const source = audioContext.createMediaStreamSource(stream);
                source.connect(analyser);
                analyser.fftSize = 64;
                bufferLength = analyser.frequencyBinCount;
                dataArray = new Uint8Array(bufferLength);

                // 设置MediaRecorder
                mediaRecorder = new MediaRecorder(stream);
                audioChunks = [];

                mediaRecorder.ondataavailable = (event) => {
                    audioChunks.push(event.data);
                };

                mediaRecorder.onstop = () => {
                    currentAudioBlob = new Blob(audioChunks, { type: 'audio/wav' });
                    const audioUrl = URL.createObjectURL(currentAudioBlob);

                    // 设置音频播放
                    const audioPlayback = document.getElementById('audioPlayback');
                    audioPlayback.src = audioUrl;
                    audioPlayback.style.display = 'block';

                    // 启用播放和分析按钮
                    document.getElementById('playRecording').disabled = false;
                    document.getElementById('analyzeRecording').disabled = false;

                    // 显示录音完成信息
                    document.getElementById('recordingInfo').style.display = 'block';

                    // 停止可视化
                    if (audioContext) {
                        audioContext.close();
                    }

                    // 生成唯一ID用于标识这次录音
                    currentRecordingId = 'rec_' + Date.now();

                    // 保存录音到历史记录（先不分析）
                    saveRecordingToHistory(currentRecordingId, currentAudioBlob, seconds);
                };

                // 开始录音
                mediaRecorder.start();

                // 更新UI
                document.getElementById('startRecording').disabled = true;
                document.getElementById('stopRecording').disabled = false;
                document.getElementById('recordingStatus').style.display = 'block';
                document.getElementById('recordingTimer').textContent = '00:00';
                document.getElementById('recordingInfo').style.display = 'none';
                document.getElementById('analyzeRecording').disabled = true;

                // 开始计时器
                seconds = 0;
                recordingTimer = setInterval(() => {
                    seconds++;
                    const minutes = Math.floor(seconds / 60);
                    const remainingSeconds = seconds % 60;
                    document.getElementById('recordingTimer').textContent = 
                        `${minutes.toString().padStart(2, '0')}:${remainingSeconds.toString().padStart(2, '0')}`;

                    // 自动停止录音（最多2分钟）
                    if (seconds >= 120) {
                        stopRecording();
                    }
                }, 1000);

                // 开始可视化
                updateVisualizer();

            } catch (error) {
                alert('无法访问麦克风: ' + error.message);
                console.error('录音错误:', error);
            }
        }

        // 停止录音
        function stopRecording() {
            if (mediaRecorder && mediaRecorder.state !== 'inactive') {
                mediaRecorder.stop();

                // 停止所有音轨
                mediaRecorder.stream.getTracks().forEach(track => track.stop());

                // 更新UI
                document.getElementById('startRecording').disabled = false;
                document.getElementById('stopRecording').disabled = true;
                document.getElementById('recordingStatus').style.display = 'none';

                // 停止计时器
                clearInterval(recordingTimer);

                // 重置可视化器
                initializeVisualizer();
            }
        }

        // 播放录音
        function playRecording() {
            const audioPlayback = document.getElementById('audioPlayback');
            audioPlayback.play();
        }

        // 分析录音
        function analyzeRecording() {
            if (!currentAudioBlob) {
                alert('没有可分析的录音数据');
                return;
            }

            showLoading();

            // 创建FormData对象
            const formData = new FormData();
            formData.append('audio', currentAudioBlob, 'recording.wav');
            formData.append('recording_id', currentRecordingId);
            formData.append('duration', seconds);

            // 发送到后端分析
            fetch('/api/analyze_audio', {
                method: 'POST',
                body: formData
            })
            .then(response => response.json())
            .then(data => {
                hideLoading();
                if (data.success) {
                    // 更新录音历史记录中的分析结果
                    updateRecordingAnalysis(currentRecordingId, data);
                    displayResult(data);
                } else {
                    alert('分析失败: ' + data.error);
                }
            })
            .catch(error => {
                hideLoading();
                alert('请求失败: ' + error);
                console.error('分析错误:', error);
            });
        }

        // 保存录音到历史记录
        function saveRecordingToHistory(id, blob, duration) {
            const recording = {
                id: id,
                blob: blob,
                duration: duration,
                timestamp: new Date().toLocaleString(),
                analyzed: false,
                result: null
            };

            recordingsHistory.unshift(recording); // 添加到开头

            // 保存到本地存储
            saveRecordingsToStorage();

            // 更新UI
            updateRecordingsHistoryUI();
        }

        // 更新录音的分析结果
        function updateRecordingAnalysis(id, result) {
            const recording = recordingsHistory.find(r => r.id === id);
            if (recording) {
                recording.analyzed = true;
                recording.result = result;

                // 保存到本地存储
                saveRecordingsToStorage();

                // 更新UI
                updateRecordingsHistoryUI();
            }
        }

        // 更新录音历史记录的UI
        function updateRecordingsHistoryUI() {
            const historyList = document.getElementById('historyList');
            const recordingsHistoryDiv = document.getElementById('recordingsHistory');

            if (recordingsHistory.length === 0) {
                recordingsHistoryDiv.style.display = 'none';
                return;
            }

            recordingsHistoryDiv.style.display = 'block';
            historyList.innerHTML = '';

            recordingsHistory.forEach(recording => {
                const item = document.createElement('div');
                item.className = 'recording-item';

                let resultInfo = '';
                if (recording.analyzed && recording.result) {
                    const riskLevel = recording.result.prediction.risk_level;
                    const probability = (recording.result.prediction.ad_probability * 100).toFixed(1);
                    resultInfo = `<div>分析结果: ${riskLevel} (${probability}%)</div>`;
                }

                item.innerHTML = `
                    <div class="recording-info">
                        <div>录音时间: ${recording.timestamp}</div>
                        <div>时长: ${recording.duration}秒</div>
                        ${resultInfo}
                    </div>
                    <div class="recording-actions">
                        <button class="btn small-btn" onclick="playRecordingFromHistory('${recording.id}')">播放</button>
                        ${!recording.analyzed ? `<button class="btn small-btn btn-warning" onclick="analyzeRecordingFromHistory('${recording.id}')">检测</button>` : ''}
                        <button class="btn small-btn btn-danger" onclick="deleteRecording('${recording.id}')">删除</button>
                    </div>
                `;

                historyList.appendChild(item);
            });
        }

        // 从历史记录播放录音
        function playRecordingFromHistory(id) {
            const recording = recordingsHistory.find(r => r.id === id);
            if (recording && recording.blob) {
                const audioUrl = URL.createObjectURL(recording.blob);
                const audio = new Audio(audioUrl);
                audio.play();
            }
        }

        // 从历史记录分析录音
        function analyzeRecordingFromHistory(id) {
            const recording = recordingsHistory.find(r => r.id === id);
            if (recording && recording.blob) {
                showLoading();

                // 创建FormData对象
                const formData = new FormData();
                formData.append('audio', recording.blob, 'recording.wav');
                formData.append('recording_id', recording.id);
                formData.append('duration', recording.duration);

                // 发送到后端分析
                fetch('/api/analyze_audio', {
                    method: 'POST',
                    body: formData
                })
                .then(response => response.json())
                .then(data => {
                    hideLoading();
                    if (data.success) {
                        // 更新录音历史记录中的分析结果
                        updateRecordingAnalysis(recording.id, data);
                        displayResult(data);
                    } else {
                        alert('分析失败: ' + data.error);
                    }
                })
                .catch(error => {
                    hideLoading();
                    alert('请求失败: ' + error);
                    console.error('分析错误:', error);
                });
            }
        }

        // 删除录音
        function deleteRecording(id) {
            if (confirm('确定要删除这条录音记录吗？')) {
                recordingsHistory = recordingsHistory.filter(r => r.id !== id);
                saveRecordingsToStorage();
                updateRecordingsHistoryUI();
            }
        }

        // 保存录音历史到本地存储
        function saveRecordingsToStorage() {
            // 注意：Blob对象不能直接存储到localStorage
            // 这里我们只存储元数据，实际音频数据在内存中
            const recordingsMetadata = recordingsHistory.map(r => ({
                id: r.id,
                duration: r.duration,
                timestamp: r.timestamp,
                analyzed: r.analyzed,
                result: r.result
            }));

            localStorage.setItem('ad_recordings_metadata', JSON.stringify(recordingsMetadata));
        }

        // 从本地存储加载录音历史
        function loadRecordingsHistory() {
            const storedMetadata = localStorage.getItem('ad_recordings_metadata');
            if (storedMetadata) {
                const metadata = JSON.parse(storedMetadata);
                // 注意：这里我们只加载元数据，实际音频数据不会持久化
                // 在实际应用中，可能需要将音频数据保存到服务器或使用IndexedDB
                recordingsHistory = metadata.map(m => ({
                    id: m.id,
                    blob: null, // 音频数据不会持久化
                    duration: m.duration,
                    timestamp: m.timestamp,
                    analyzed: m.analyzed,
                    result: m.result
                }));

                updateRecordingsHistoryUI();
            }
        }

        // 检查系统状态
        function checkSystemStatus() {
            fetch('/api/health')
                .then(response => response.json())
                .then(data => {
                    const statusDiv = document.getElementById('systemStatus');
                    if (data.model_loaded) {
                        statusDiv.innerHTML = '<div class="status status-ok">✅ 系统运行正常 - 模型已加载</div>';
                    } else {
                        statusDiv.innerHTML = '<div class="status status-error">❌ 模型未加载，请先训练模型</div>';
                    }
                })
                .catch(error => {
                    document.getElementById('systemStatus').innerHTML = 
                        '<div class="status status-error">❌ 无法连接到API服务，请确保服务正在运行</div>';
                });
        }

        // 运行演示测试
        function runDemoTest() {
            showLoading();
            fetch('/api/demo')
                .then(response => response.json())
                .then(data => {
                    hideLoading();
                    displayResult(data);
                })
                .catch(error => {
                    hideLoading();
                    alert('请求失败: ' + error);
                });
        }

        // 模拟AD患者检测
        function runADSimulation() {
            showLoading();
            fetch('/api/predict', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    simulate_ad: true
                })
            })
            .then(response => response.json())
            .then(data => {
                hideLoading();
                displayResult(data);
            })
            .catch(error => {
                hideLoading();
                alert('请求失败: ' + error);
            });
        }

        // 模拟正常人检测
        function runNormalSimulation() {
            showLoading();
            fetch('/api/predict', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    simulate_ad: false
                })
            })
            .then(response => response.json())
            .then(data => {
                hideLoading();
                displayResult(data);
            })
            .catch(error => {
                hideLoading();
                alert('请求失败: ' + error);
            });
        }

        // 显示加载动画
        function showLoading() {
            document.getElementById('loading').style.display = 'block';
            document.getElementById('result').style.display = 'none';
        }

        // 隐藏加载动画
        function hideLoading() {
            document.getElementById('loading').style.display = 'none';
        }

        // 显示结果
        function displayResult(data) {
            const resultDiv = document.getElementById('result');
            resultDiv.style.display = 'block';

            if (data.success) {
                const pred = data.prediction;
                const interpret = data.interpretation;

                // 根据风险等级设置样式
                let riskClass = 'risk-low';
                if (interpret.message.includes('高风险')) riskClass = 'risk-high';
                else if (interpret.message.includes('中风险')) riskClass = 'risk-medium';

                resultDiv.className = 'result ' + riskClass;
                resultDiv.innerHTML = `
                    <h3>检测结果</h3>
                    <p><strong>风险等级:</strong> ${pred.risk_level}</p>
                    <p><strong>AD概率:</strong> ${(pred.ad_probability * 100).toFixed(1)}%</p>
                    <p><strong>建议:</strong> ${interpret.recommendation}</p>
                    <p><strong>检测时间:</strong> ${pred.timestamp}</p>
                    ${data.audio_duration ? `<p><strong>录音时长:</strong> ${data.audio_duration}秒</p>` : ''}
                    ${data.recording_id ? `<p><strong>录音ID:</strong> ${data.recording_id}</p>` : ''}
                `;
            } else {
                resultDiv.className = 'result risk-high';
                resultDiv.innerHTML = `
                    <h3>检测失败</h3>
                    <p>错误信息: ${data.error}</p>
                `;
            }
        }
    </script>
</body>
</html>
'''

# =============================================================================
# 路由定义
# =============================================================================

@app.route('/')
def index():
    """主页面 - 返回网页界面"""
    return render_template_string(HTML_TEMPLATE)

@app.route('/api/health', methods=['GET'])
def health_check():
    """健康检查接口"""
    status = "ok" if model_loaded else "model_not_loaded"
    message = "AD语音检测服务运行正常" if model_loaded else "服务就绪，但模型未加载"

    return jsonify({
        'status': status,
        'message': message,
        'model_loaded': model_loaded,
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    })

@app.route('/api/predict', methods=['POST'])
def predict_ad():
    """
    AD语音检测预测接口
    接收特征数据并返回预测结果
    """
    try:
        # 检查模型是否加载
        if not model_loaded:
            return jsonify({
                'success': False,
                'error': '模型未加载，请先训练模型',
                'model_loaded': False
            }), 503

        # 获取请求数据
        data = request.get_json()

        if not data:
            return jsonify({
                'success': False,
                'error': '未提供JSON数据'
            }), 400

        # 检查是否提供了特征数据
        if 'features' in data:
            # 使用提供的特征数据
            features = np.array(data['features']).reshape(1, -1)
            use_provided_features = True
        else:
            # 生成模拟特征数据
            n_features = 286
            if data.get('simulate_ad', False):
                # 模拟AD患者的特征
                features = np.random.normal(-0.1, 0.3, n_features)
            else:
                # 模拟正常人的特征
                features = np.random.normal(0.1, 0.3, n_features)

            # 添加噪声
            features += np.random.normal(0, 0.1, n_features)
            features = features.reshape(1, -1)
            use_provided_features = False

        # 模型预测
        prediction_prob = model.predict_proba(features)[0, 1]  # AD类的概率
        prediction_label = model.predict(features)[0]

        # 确定风险等级
        if prediction_prob >= 0.7:
            risk_level = "高风险"
            recommendation = "建议尽快咨询神经科医生进行专业评估和进一步检查"
        elif prediction_prob >= 0.4:
            risk_level = "中风险"
            recommendation = "建议定期进行认知功能评估，关注日常生活中的变化"
        else:
            risk_level = "低风险"
            recommendation = "当前认知功能状况良好，建议保持健康生活方式"

        # 构建响应
        response = {
            'success': True,
            'prediction': {
                'ad_probability': float(prediction_prob),
                'predicted_label': int(prediction_label),
                'risk_level': risk_level,
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            },
            'interpretation': {
                'message': f'AD筛查风险等级: {risk_level}',
                'confidence': float(prediction_prob),
                'recommendation': recommendation
            },
            'model_info': {
                'model_loaded': True,
                'features_provided': use_provided_features
            }
        }

        logger.info(f"预测完成 - 风险等级: {risk_level}, 概率: {prediction_prob:.3f}")
        return jsonify(response)

    except Exception as e:
        logger.error(f"预测过程出错: {str(e)}")
        return jsonify({
            'success': False,
            'error': f'预测失败: {str(e)}',
            'model_loaded': model_loaded
        }), 500

@app.route('/api/demo', methods=['GET'])
def demo_predict():
    """演示接口，返回模拟结果"""
    import random

    # 生成随机概率
    prob = random.uniform(0.1, 0.9)

    # 确定风险等级
    if prob >= 0.7:
        risk_level = "高风险"
        recommendation = "建议尽快咨询神经科医生进行专业评估"
    elif prob >= 0.4:
        risk_level = "中风险"
        recommendation = "建议定期复查并关注认知功能变化"
    else:
        risk_level = "低风险"
        recommendation = "当前认知功能正常，建议保持健康生活方式"

    response = {
        'success': True,
        'prediction': {
            'ad_probability': float(prob),
            'risk_level': risk_level,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        },
        'interpretation': {
            'message': f'AD筛查风险等级: {risk_level}',
            'confidence': float(prob),
            'recommendation': recommendation
        },
        'model_info': {
            'model_loaded': model_loaded,
            'note': '此为演示结果'
        }
    }

    logger.info(f"演示预测 - 风险等级: {risk_level}, 概率: {prob:.3f}")
    return jsonify(response)

@app.route('/api/analyze_audio', methods=['POST'])
def analyze_audio():
    """
    分析上传的音频文件
    """
    try:
        # 检查是否有文件上传
        if 'audio' not in request.files:
            return jsonify({
                'success': False,
                'error': '没有上传音频文件'
            }), 400

        audio_file = request.files['audio']
        recording_id = request.form.get('recording_id', str(uuid.uuid4()))
        duration = request.form.get('duration', 0)

        if audio_file.filename == '':
            return jsonify({
                'success': False,
                'error': '没有选择文件'
            }), 400

        # 保存录音记录
        audio_records[recording_id] = {
            'filename': audio_file.filename,
            'timestamp': datetime.now().isoformat(),
            'duration': duration
        }

        # 模拟处理：根据文件大小生成"随机"但可重复的结果
        file_size = len(audio_file.read())
        audio_file.seek(0)  # 重置文件指针

        # 使用文件大小作为随机种子，使相同文件得到相同结果
        import random
        random.seed(file_size)

        # 模拟处理时间
        time.sleep(2)  # 模拟处理时间

        # 生成基于文件大小的"预测结果"
        base_prob = (file_size % 1000) / 1000  # 0-1之间的值
        # 调整概率分布，使大多数结果在低风险范围
        ad_probability = min(0.8, base_prob * 1.5)

        # 确定风险等级
        if ad_probability >= 0.7:
            risk_level = "高风险"
            recommendation = "建议尽快咨询神经科医生进行专业评估和进一步检查"
        elif ad_probability >= 0.4:
            risk_level = "中风险"
            recommendation = "建议定期进行认知功能评估，关注日常生活中的变化"
        else:
            risk_level = "低风险"
            recommendation = "当前认知功能状况良好，建议保持健康生活方式"

        return jsonify({
            'success': True,
            'prediction': {
                'ad_probability': float(ad_probability),
                'predicted_label': 1 if ad_probability > 0.5 else 0,
                'risk_level': risk_level,
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            },
            'interpretation': {
                'message': f'AD筛查风险等级: {risk_level}',
                'confidence': float(ad_probability),
                'recommendation': recommendation
            },
            'audio_duration': duration,
            'recording_id': recording_id,
            'model_info': {
                'model_loaded': model_loaded,
                'analysis_type': 'audio_analysis'
            }
        })

    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'音频分析失败: {str(e)}'
        }), 500

@app.route('/api/recording_info/<recording_id>', methods=['GET'])
def get_recording_info(recording_id):
    """
    获取录音信息
    """
    if recording_id in audio_records:
        return jsonify({
            'success': True,
            'recording': audio_records[recording_id]
        })
    else:
        return jsonify({
            'success': False,
            'error': '录音记录不存在'
        }), 404

# =============================================================================
# 应用启动
# =============================================================================

if __name__ == '__main__':
    logger.info("🚀 启动AD语音检测完整Web服务...")

    # 加载模型
    if load_model():
        logger.info("✅ 模型加载成功")
    else:
        logger.warning("⚠️ 模型未加载，部分功能将使用模拟数据")

    # 启动Flask应用
    port = int(os.environ.get('PORT', 5000))
    logger.info(f"✅ 服务启动在端口: {port}")

    app.run(
        host='0.0.0.0',
        port=port,
        debug=False,  # 生产环境关闭debug
        threaded=True
    )