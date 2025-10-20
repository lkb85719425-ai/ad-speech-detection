"""
AD语音检测API服务
为微信小程序提供模型预测接口
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import joblib
import os
from datetime import datetime
import logging

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)  # 允许跨域请求

# 全局变量存储模型
model = None
model_loaded = False


def load_model():
    """加载训练好的模型"""
    global model, model_loaded

    try:
        # 尝试不同的可能路径
        possible_paths = [
            'ad_speech_detector.pkl',  # 当前目录
            './ad_speech_detector.pkl',
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


if __name__ == '__main__':
    logger.info("🚀 启动AD语音检测API服务...")

    # 加载模型
    if load_model():
        # 启动Flask应用
        logger.info("✅ 服务启动成功，正在监听端口 5000")
        logger.info("📊 可用接口:")
        logger.info("  GET  /api/health     - 健康检查")
        logger.info("  GET  /api/demo       - 演示预测")
        logger.info("  POST /api/predict    - 模型预测")

        app.run(
            host='0.0.0.0',  # 允许外部访问
            port=5000,  # 端口号
            debug=True,  # 调试模式
            threaded=True  # 多线程
        )
    else:
        logger.error("❌ 服务启动失败：模型加载错误")