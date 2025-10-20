# 阿尔茨海默症语音检测系统

## 项目描述
基于语音特征和机器学习的光GBM分类器的AD检测系统

## 文件说明
- `index.html` - 在线演示版本（静态网站）
- `train_model.py` - 模型训练代码
- `api_service.py` - API服务代码  
- `web_interface.py` - 原始网页界面代码

## 在线演示
访问：https://[您的用户名].github.io/ad-speech-detection/

## 本地运行完整版本
1. 安装依赖：`pip install -r requirements.txt`
2. 训练模型：`python train_model.py`
3. 启动API：`python api_service.py`
4. 启动网页：`python web_interface.py`
5. 访问：http://localhost:8080

## 技术栈
- Python + Flask
- LightGBM机器学习
- 语音特征提取
- Web Audio API
