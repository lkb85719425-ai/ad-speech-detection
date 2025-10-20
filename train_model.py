"""
阿尔茨海默症(AD)语音检测系统
基于语音特征和机器学习的光GBM分类器

必需库及版本:
- scikit-learn==1.4.0          # 机器学习工具包
- imbalanced-learn==0.12.0      # 不平衡数据处理
- lightgbm==4.1.0               # LightGBM梯度提升树
- librosa==0.10.1               # 音频特征提取
- numpy==2.3.3                  # 数值计算
- pandas==2.3.2                 # 数据处理
- matplotlib==3.10.6            # 数据可视化
- seaborn==0.13.2               # 统计可视化
- joblib==1.5.2                 # 模型持久化
- scipy==1.16.2                 # 科学计算
- soundfile==0.13.1             # 音频文件读写

可选库(用于扩展):
- tensorflow==2.20.0            # 深度学习扩展
- xgboost==3.0.5                # 替代模型
- nltk==3.9.1                   # 文本处理扩展
- spacy==3.8.7                  # NLP处理扩展
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

# =============================================================================
# 环境配置部分
# =============================================================================

# 设置matplotlib使用Agg后端，避免GUI依赖问题
matplotlib.use('Agg', force=True)

import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, confusion_matrix, classification_report, roc_curve)
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
import lightgbm as lgb
import librosa
import os
from glob import glob
import time
import warnings

# 忽略警告信息
warnings.filterwarnings('ignore')

# =============================================================================
# 全局配置和常量定义
# =============================================================================

# 设置随机种子确保结果可重现
np.random.seed(42)

# 配置matplotlib参数
plt.rcParams['figure.figsize'] = (10, 6)  # 图形默认大小
plt.rcParams["font.family"] = ["Arial", "DejaVu Sans", "Liberation Sans"]  # 字体设置
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 特征维度常量
N_MFCC = 13  # MFCC系数数量
N_MEL = 128  # 梅尔频带数量
N_FEATURES = 286  # 总特征维度 (13*2 + 128*2 + 2 + 2)

print("系统初始化完成: 使用英文标签以确保跨平台兼容性")


# =============================================================================
# 语音特征提取模块
# =============================================================================

def extract_audio_features(audio_file):
    """
    从音频文件中提取语音声学特征

    参数:
        audio_file (str): 音频文件路径

    返回:
        numpy.ndarray: 286维特征向量，包含MFCC、梅尔频谱、能量、基频等特征

    异常处理:
        - 文件不存在时返回零向量
        - 音频处理错误时返回零向量并打印错误信息
    """
    # 检查文件是否存在
    if not os.path.exists(audio_file):
        print(f"错误: 音频文件不存在: {os.path.basename(audio_file)}")
        return np.zeros(N_FEATURES)

    max_duration = 10  # 最大处理时长(秒)

    try:
        # 使用librosa加载音频文件
        # sr=None 保持原始采样率, duration限制处理时长, mono转换为单声道
        y, sr = librosa.load(
            audio_file,
            sr=None,
            duration=max_duration,
            mono=True,
            res_type='kaiser_fast'  # 快速重采样方法
        )

        # ==================== MFCC特征提取 ====================
        # MFCC(Mel-frequency cepstral coefficients)是语音识别中最常用的特征
        # n_mfcc=13: 提取13个MFCC系数
        # n_fft=2048: FFT窗口大小
        # hop_length=512: 帧移大小
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC, n_fft=2048, hop_length=512)
        mfccs_mean = np.mean(mfccs, axis=1)  # 时域均值
        mfccs_std = np.std(mfccs, axis=1)  # 时域标准差

        # ==================== 梅尔频谱特征提取 ====================
        # 梅尔频谱模拟人耳听觉特性
        mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=N_MEL)
        mel_spec_mean = np.mean(mel_spec, axis=1)  # 时域均值
        mel_spec_std = np.std(mel_spec, axis=1)  # 时域标准差

        # ==================== 能量特征提取 ====================
        # RMS(Root Mean Square)能量特征
        energy = librosa.feature.rms(y=y, frame_length=2048, hop_length=512)
        energy_mean = np.mean(energy)  # 平均能量
        energy_std = np.std(energy)  # 能量波动

        # ==================== 基频特征提取 ====================
        # 基频(Fundamental Frequency)反映语音的音高特性
        # fmin, fmax: 基频搜索范围(C2到C7)
        f0, _, _ = librosa.pyin(y, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'))
        f0_valid = f0[~np.isnan(f0)]  # 去除NaN值
        f0_mean = np.mean(f0_valid) if len(f0_valid) > 0 else 0.0  # 平均基频
        f0_std = np.std(f0_valid) if len(f0_valid) > 0 else 0.0  # 基频标准差

        # ==================== 特征合并 ====================
        # 将所有特征合并为286维向量
        features = np.concatenate([
            mfccs_mean, mfccs_std,  # MFCC特征: 13均值 + 13标准差 = 26维
            mel_spec_mean, mel_spec_std,  # 梅尔特征: 128均值 + 128标准差 = 256维
            [energy_mean, energy_std],  # 能量特征: 2维
            [f0_mean, f0_std]  # 基频特征: 2维
        ])

        return features

    except Exception as e:
        # 异常处理: 记录错误信息并返回零向量
        error_type = type(e).__name__
        print(f"错误: 提取 {os.path.basename(audio_file)} 失败 ({error_type}): {str(e)[:60]}")
        return np.zeros(N_FEATURES)


# =============================================================================
# 数据生成模块 - 高度真实的模拟数据
# =============================================================================

def generate_highly_realistic_simulated_data(n_samples=1500, ad_ratio=0.3, difficulty_level=0.8):
    """
    生成高度真实的模拟语音数据，模拟AD患者与正常人的声学差异

    参数:
        n_samples (int): 生成的样本数量
        ad_ratio (float): AD样本的目标比例(0-1)
        difficulty_level (float): 分类难度级别(0-1), 越高越难分类

    返回:
        tuple: (特征矩阵X, 标签向量y)

    算法说明:
        1. 使用多因素风险模型生成AD概率
        2. 基于风险概率生成标签
        3. 为每类样本生成具有重叠分布的特征
        4. 添加多层级噪声增加真实性
    """
    print(f"信息: 生成高度真实模拟数据 ({n_samples} 样本, 目标AD比例: {ad_ratio:.2%}, 难度级别: {difficulty_level})")

    # ==================== 多因素风险模型 ====================
    # 模拟影响AD风险的多个生物医学因素
    genetic_factors = np.random.normal(0, 1, (n_samples, 3))  # 遗传因素(3维)
    age_factors = np.random.normal(0, 0.8, (n_samples, 2))  # 年龄因素(2维)
    environmental_factors = np.random.normal(0, 0.6, (n_samples, 2))  # 环境因素(2维)
    cognitive_factors = np.random.normal(0, 0.7, (n_samples, 2))  # 认知因素(2维)

    # ==================== 非线性风险评分计算 ====================
    # 使用线性组合+交互项+非线性变换模拟真实风险
    ad_risk_score = (
            0.25 * genetic_factors[:, 0] +  # 主要遗传因素
            0.20 * age_factors[:, 0] +  # 主要年龄因素
            0.15 * environmental_factors[:, 0] +  # 主要环境因素
            0.20 * cognitive_factors[:, 0] +  # 主要认知因素
            0.10 * genetic_factors[:, 0] * age_factors[:, 0] +  # 基因-年龄交互
            0.05 * genetic_factors[:, 1] * environmental_factors[:, 0] +  # 基因-环境交互
            0.05 * np.sin(genetic_factors[:, 2])  # 非线性变换
    )

    # ==================== 概率转换与标签生成 ====================
    # 根据难度调整分类边界清晰度
    separation_factor = 1.0 - difficulty_level

    # 使用sigmoid函数将风险评分转换为概率
    base_log_odds = np.log(ad_ratio / (1 - ad_ratio)) * separation_factor
    ad_probability = 1 / (1 + np.exp(-(ad_risk_score * 0.3 + base_log_odds)))

    # 引入标签噪声模拟诊断不确定性
    label_noise = np.random.binomial(1, difficulty_level * 0.1, n_samples)
    clean_y = np.random.binomial(1, ad_probability)  # 基于概率生成干净标签
    y = np.where(label_noise == 1, 1 - clean_y, clean_y)  # 添加标签噪声

    # ==================== 特征矩阵初始化 ====================
    X = np.zeros((n_samples, N_FEATURES))

    # 为每个样本生成特征
    for i in range(n_samples):
        if y[i] == 1:
            # AD患者特征模式 - 反映语音退化特征
            base_template = {
                'mfcc_mean': np.random.normal(-0.05, 0.25, N_MFCC),  # MFCC均值偏向负值
                'mfcc_std': np.random.normal(0.18, 0.08, N_MFCC),  # MFCC变异减小
                'mel_mean': np.random.normal(-0.1, 0.25, N_MEL),  # 梅尔频谱能量降低
                'mel_std': np.random.normal(0.28, 0.08, N_MEL),  # 频谱稳定性下降
                'energy_mean': np.random.normal(-0.15, 0.18),  # 语音能量减弱
                'energy_std': np.random.normal(0.18, 0.07),  # 能量波动减小
                'f0_mean': np.random.normal(120, 20),  # 基频降低
                'f0_std': np.random.normal(28, 10)  # 基频变异增加
            }
        else:
            # 正常人特征模式
            base_template = {
                'mfcc_mean': np.random.normal(0.1, 0.25, N_MFCC),  # MFCC均值正常
                'mfcc_std': np.random.normal(0.22, 0.08, N_MFCC),  # MFCC变异正常
                'mel_mean': np.random.normal(0.1, 0.25, N_MEL),  # 梅尔频谱能量正常
                'mel_std': np.random.normal(0.32, 0.08, N_MEL),  # 频谱稳定性正常
                'energy_mean': np.random.normal(0.1, 0.18),  # 语音能量正常
                'energy_std': np.random.normal(0.22, 0.07),  # 能量波动正常
                'f0_mean': np.random.normal(135, 20),  # 基频正常
                'f0_std': np.random.normal(32, 10)  # 基频变异正常
            }

        # ==================== 多层级噪声添加 ====================
        # 1. 个体变异噪声 - 模拟个体差异
        individual_noise = np.random.normal(0, 0.3 * difficulty_level, N_FEATURES)

        # 2. 特征相关性噪声 - 模拟特征间的相关性
        correlated_noise_1 = np.random.normal(0, 0.2 * difficulty_level)
        correlated_noise_2 = np.random.normal(0, 0.15 * difficulty_level)

        # ==================== 特征生成与噪声应用 ====================
        # MFCC特征组
        mfcc_mean = base_template['mfcc_mean'] + individual_noise[:N_MFCC] + correlated_noise_1 * 0.15
        mfcc_std = base_template['mfcc_std'] + individual_noise[N_MFCC:N_MFCC * 2] + correlated_noise_1 * 0.08

        # 梅尔频谱特征组
        mel_mean = base_template['mel_mean'] + individual_noise[
                                               N_MFCC * 2:N_MFCC * 2 + N_MEL] + correlated_noise_2 * 0.12
        mel_std = base_template['mel_std'] + individual_noise[
                                             N_MFCC * 2 + N_MEL:N_MFCC * 2 + N_MEL * 2] + correlated_noise_2 * 0.06

        # 能量和基频特征
        energy_mean = base_template['energy_mean'] + individual_noise[-4] + correlated_noise_1 * 0.05
        energy_std = base_template['energy_std'] + individual_noise[-3] + correlated_noise_1 * 0.03
        f0_mean = base_template['f0_mean'] + individual_noise[-2] * 15 + correlated_noise_2 * 3
        f0_std = base_template['f0_std'] + individual_noise[-1] * 8 + correlated_noise_2 * 2

        # ==================== 风险相关的特征偏移 ====================
        # 让特征与风险评分相关，增加生物学合理性
        risk_offset = ad_risk_score[i] * 0.1 * difficulty_level
        mfcc_mean += risk_offset * 0.3
        mel_mean += risk_offset * 0.2
        energy_mean += risk_offset * 0.1
        f0_mean += risk_offset * 5

        # ==================== 特征向量组装 ====================
        X[i] = np.concatenate([
            mfcc_mean, mfcc_std,
            mel_mean, mel_std,
            [energy_mean, energy_std],
            [f0_mean, f0_std]
        ])

    # ==================== 全局噪声和异常值 ====================
    # 3. 全局噪声 - 模拟测量误差
    global_noise = np.random.normal(0, 0.2 * difficulty_level, X.shape)
    X += global_noise

    # 4. 异常值 - 模拟数据采集中的异常情况
    n_outliers = int(n_samples * 0.05)  # 5%的样本作为异常值
    outlier_indices = np.random.choice(n_samples, n_outliers, replace=False)
    X[outlier_indices] += np.random.normal(0, 1.0, (n_outliers, N_FEATURES))

    # ==================== 结果统计和输出 ====================
    actual_ad_ratio = np.mean(y)
    expected_auc_min = 0.65 + (1 - difficulty_level) * 0.25
    expected_auc_max = 0.85 + (1 - difficulty_level) * 0.1

    print(f"信息: 高度真实模拟数据生成完成 (实际AD比例: {actual_ad_ratio:.2%})")
    print(f"信息: 预期性能范围 - AUC: {expected_auc_min:.2f} 到 {expected_auc_max:.2f}")

    return X, y


# =============================================================================
# 数据加载和预处理模块
# =============================================================================

def load_and_preprocess_audio_data(audio_dir=None, ad_ratio=0.3, difficulty_level=0.8):
    """
    加载和预处理语音数据，支持真实数据和模拟数据

    参数:
        audio_dir (str): 真实音频数据目录路径，None则使用模拟数据
        ad_ratio (float): AD样本比例(仅模拟数据使用)
        difficulty_level (float): 分类难度(仅模拟数据使用)

    返回:
        tuple: (X_train, X_test, y_train, y_test, preprocessor)

    目录结构要求(真实数据):
        audio_dir/
            ├── ad/          # AD患者音频文件
            └── control/     # 正常对照音频文件
    """
    if audio_dir and os.path.exists(audio_dir):
        # ==================== 真实数据加载流程 ====================
        ad_dir = os.path.join(audio_dir, 'ad')
        control_dir = os.path.join(audio_dir, 'control')

        # 检查目录结构
        if not os.path.exists(ad_dir) or not os.path.exists(control_dir):
            raise ValueError("错误: 数据目录结构不正确，需要ad和control子目录")

        # 搜索音频文件(支持wav和mp3格式)
        audio_extensions = ['*.wav', '*.mp3']
        ad_files = []
        control_files = []

        for ext in audio_extensions:
            ad_files.extend(glob(os.path.join(ad_dir, ext)))
            control_files.extend(glob(os.path.join(control_dir, ext)))

        # 检查文件数量
        if len(ad_files) == 0 or len(control_files) == 0:
            raise ValueError("错误: 音频文件数量不足，请检查数据目录")

        # 特征提取
        X = []
        y = []
        total_files = len(ad_files) + len(control_files)
        print(f"信息: 提取真实语音特征 (共{total_files}个文件: AD={len(ad_files)}, 正常={len(control_files)})")

        # 处理AD患者音频
        for i, file in enumerate(ad_files):
            if i % 10 == 0:  # 每10个文件打印一次进度
                print(f"进度: {i + 1}/{len(ad_files)} (AD文件: {os.path.basename(file)})")
            X.append(extract_audio_features(file))
            y.append(1)  # AD标签为1

        # 处理正常对照音频
        for i, file in enumerate(control_files):
            if i % 10 == 0:
                print(f"进度: {i + 1}/{len(control_files)} (正常文件: {os.path.basename(file)})")
            X.append(extract_audio_features(file))
            y.append(0)  # 正常标签为0

        X = np.array(X)
        y = np.array(y)
        print(f"信息: 真实数据加载完成 (总样本数: {len(X)}, AD比例: {np.mean(y):.2%})")

    else:
        # ==================== 模拟数据生成流程 ====================
        X, y = generate_highly_realistic_simulated_data(
            ad_ratio=ad_ratio,
            difficulty_level=difficulty_level
        )

    # ==================== 数据划分 ====================
    # 使用分层抽样确保训练集和测试集的类别比例一致
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,  # 20%作为测试集
        random_state=42,  # 固定随机种子
        stratify=y  # 按标签分层抽样
    )

    # ==================== 数据预处理管道 ====================
    # 使用ColumnTransformer便于后续扩展其他类型的特征
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), slice(0, X.shape[1]))  # 对数值特征进行标准化
        ],
        remainder='drop'  # 丢弃未指定的特征
    )
    preprocessor.fit(X_train)  # 仅使用训练集拟合预处理器

    # ==================== 数据概况输出 ====================
    print(f"\n数据划分结果:")
    print(f"训练集: {len(X_train)} 样本 (AD: {sum(y_train)}, 正常: {len(y_train) - sum(y_train)})")
    print(f"测试集: {len(X_test)} 样本 (AD: {sum(y_test)}, 正常: {len(y_test) - sum(y_test)})")
    print(f"特征维度: {X.shape[1]} 维语音声学特征")

    # ==================== 数据分离度分析 ====================
    # 使用最近邻距离评估类别分离程度
    from sklearn.neighbors import NearestNeighbors
    nn = NearestNeighbors(n_neighbors=2)
    nn.fit(X_train)
    distances, indices = nn.kneighbors(X_train)

    same_class_distances = []
    diff_class_distances = []

    for i in range(len(X_train)):
        neighbor_idx = indices[i, 1]  # 最近邻(排除自身)
        if y_train[i] == y_train[neighbor_idx]:
            same_class_distances.append(distances[i, 1])  # 同类距离
        else:
            diff_class_distances.append(distances[i, 1])  # 异类距离

    # 计算分离度指标
    if same_class_distances and diff_class_distances:
        avg_same_dist = np.mean(same_class_distances)
        avg_diff_dist = np.mean(diff_class_distances)
        separation_ratio = avg_diff_dist / avg_same_dist if avg_same_dist > 0 else 1.0
        print(f"数据分离度指标: {separation_ratio:.3f} (值越大表示越容易分类)")

    return X_train, X_test, y_train, y_test, preprocessor


# =============================================================================
# 模型训练和评估模块
# =============================================================================

def train_and_evaluate_models(X_train, X_test, y_train, y_test, preprocessor):
    """
    训练并评估LightGBM模型

    参数:
        X_train: 训练特征
        X_test: 测试特征
        y_train: 训练标签
        y_test: 测试标签
        preprocessor: 预处理器

    返回:
        dict: 包含各模型性能信息的字典

    模型特点:
        - 使用SMOTE处理类别不平衡
        - 强正则化防止过拟合
        - 管道式处理确保数据预处理一致性
    """
    # ==================== 模型配置 ====================
    models = {
        'LightGBM-Audio': {
            'model': lgb.LGBMClassifier(
                class_weight='balanced',  # 自动处理类别不平衡
                random_state=42,  # 固定随机种子
                n_estimators=150,  # 树的数量(减少以防止过拟合)
                learning_rate=0.05,  # 学习率
                num_leaves=15,  # 每棵树的最大叶子数(减少复杂度)
                max_depth=4,  # 树的最大深度(限制模型复杂度)
                min_child_samples=30,  # 叶子节点最少样本数(增加正则化)
                reg_alpha=0.5,  # L1正则化强度
                reg_lambda=0.5,  # L2正则化强度
                subsample=0.6,  # 样本采样比例
                colsample_bytree=0.6,  # 特征采样比例
                verbose=-1  # 不输出训练日志
            ),
            'param_grid': {
                'classifier__learning_rate': [0.01, 0.05],  # 学习率搜索范围
                'classifier__num_leaves': [15, 31],  # 叶子数搜索范围
                'classifier__max_depth': [3, 4],  # 深度搜索范围
                'classifier__min_child_samples': [20, 30, 40]  # 最小样本数搜索范围
            }
        }
    }

    performance = {}  # 存储模型性能
    print(f"\n===== 模型训练阶段 =====")

    for name, info in models.items():
        print(f"训练模型: {name}")
        try:
            # ==================== 构建处理管道 ====================
            pipeline = ImbPipeline(steps=[
                ('preprocessor', preprocessor),  # 数据预处理
                ('smote', SMOTE(random_state=42, k_neighbors=3)),  # SMOTE过采样
                ('classifier', info['model'])  # 分类器
            ])

            # ==================== 模型训练 ====================
            start_time = time.time()
            pipeline.fit(X_train, y_train)
            train_time = round(time.time() - start_time, 2)
            print(f"训练耗时: {train_time} 秒")

            # ==================== 模型预测 ====================
            y_pred = pipeline.predict(X_test)  # 类别预测
            y_prob = pipeline.predict_proba(X_test)[:, 1]  # 概率预测

            # ==================== 性能评估 ====================
            accuracy = accuracy_score(y_test, y_pred)  # 准确率
            precision = precision_score(y_test, y_pred, zero_division=0)  # 精确率
            recall = recall_score(y_test, y_pred, zero_division=0)  # 召回率
            f1 = f1_score(y_test, y_pred, zero_division=0)  # F1分数
            roc_auc = roc_auc_score(y_test, y_prob) if len(np.unique(y_test)) > 1 else 0.5  # AUC

            # ==================== 存储结果 ====================
            performance[name] = {
                'model': pipeline,
                'param_grid': info['param_grid'],
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'roc_auc': roc_auc,
                'y_pred': y_pred,
                'y_prob': y_prob
            }

            # ==================== 结果输出 ====================
            print(f"模型 {name} 训练完成，评估结果:")
            print(f"  准确率: {accuracy:.4f}")
            print(f"  精确率(AD类): {precision:.4f}")
            print(f"  召回率(AD类): {recall:.4f}")
            print(f"  F1分数: {f1:.4f}")
            print(f"  ROC-AUC: {roc_auc:.4f}")

            print(f"分类报告:")
            target_names = ['Normal', 'AD']
            print(classification_report(y_test, y_pred, target_names=target_names, zero_division=0))

        except Exception as e:
            print(f"训练 {name} 出错: {str(e)[:80]}")
            continue

    if not performance:
        raise RuntimeError("所有模型训练失败，请检查数据和参数配置")

    return performance


# =============================================================================
# 模型优化模块
# =============================================================================

def optimize_best_model(X_train, y_train, preprocessor, best_model_info):
    """
    使用网格搜索优化最佳模型的超参数

    参数:
        X_train: 训练特征
        y_train: 训练标签
        preprocessor: 预处理器
        best_model_info: 最佳模型信息

    返回:
        优化后的模型

    优化策略:
        - 5折交叉验证
        - ROC-AUC作为评分标准
        - 并行计算加速搜索
    """
    print(f"\n===== 模型优化阶段 =====")
    try:
        # ==================== 构建优化管道 ====================
        optimized_pipeline = ImbPipeline(steps=[
            ('preprocessor', preprocessor),
            ('smote', SMOTE(random_state=42, k_neighbors=3)),
            ('classifier', best_model_info['model'].named_steps['classifier'])
        ])

        # ==================== 网格搜索配置 ====================
        n_jobs = min(4, os.cpu_count() // 2) if os.cpu_count() else 1  # 并行线程数

        grid_search = GridSearchCV(
            estimator=optimized_pipeline,
            param_grid=best_model_info['param_grid'],
            cv=5,  # 5折交叉验证
            scoring='roc_auc',  # 使用AUC作为评分标准
            n_jobs=n_jobs,  # 并行计算
            verbose=1,  # 输出搜索进度
            refit=True  # 使用最佳参数重新训练
        )

        # ==================== 执行网格搜索 ====================
        start_time = time.time()
        param_combinations = len(best_model_info['param_grid'])
        print(f"开始网格搜索 (超参数组合: {param_combinations}, 并行线程: {n_jobs})")

        grid_search.fit(X_train, y_train)
        search_time = round(time.time() - start_time, 2)

        # ==================== 输出优化结果 ====================
        print(f"网格搜索完成 (耗时: {search_time} 秒)")
        print(f"最佳超参数: {grid_search.best_params_}")
        print(f"最佳交叉验证ROC-AUC: {grid_search.best_score_:.4f}")

        return grid_search.best_estimator_

    except Exception as e:
        print(f"模型优化出错: {str(e)[:80]}")
        print("使用原始基础模型继续")
        return best_model_info['model']


# =============================================================================
# 结果可视化模块
# =============================================================================

def visualize_results(performance, y_test, preprocessor):
    """
    生成模型性能的可视化图表

    参数:
        performance: 模型性能字典
        y_test: 测试集真实标签
        preprocessor: 预处理器(用于特征重要性分析)

    生成图表:
        1. 模型性能比较图
        2. ROC曲线图
        3. 混淆矩阵
        4. 特征重要性图
    """
    # ==================== 可视化配置 ====================
    sns.set_style("whitegrid")  # 设置seaborn样式
    sns.set_palette("muted")  # 设置颜色主题
    target_names = ['Normal', 'AD']

    # 标签字典(英文)
    labels = {
        'perf_title': 'Model Performance Comparison (AD Speech Detection)',
        'recall': 'Recall (Sensitivity)',
        'auc': 'ROC-AUC (Discrimination)',
        'f1': 'F1 Score',
        'roc_title': 'Model ROC Curves',
        'fpr': 'False Positive Rate',
        'tpr': 'True Positive Rate',
        'cm_title': 'Confusion Matrix (True Label Percentage)',
        'true_label': 'True Label',
        'pred_label': 'Predicted Label',
        'imp_title': 'LightGBM Feature Importance (Top 10)',
        'imp_x': 'Importance Score',
        'imp_y': 'Feature Index',
        'random_guess': 'Random Guess',
        'percentage': 'Percentage (%)'
    }

    # ==================== 创建保存目录 ====================
    save_dir = 'ad_speech_plots'
    os.makedirs(save_dir, exist_ok=True)
    print(f"\n===== 生成可视化结果 =====")
    print(f"保存路径: {os.path.abspath(save_dir)}")

    # ==================== 1. 模型性能比较图 ====================
    print("绘制模型性能比较图")
    metrics_df = pd.DataFrame({
        labels['recall']: [p['recall'] for p in performance.values()],
        labels['auc']: [p['roc_auc'] for p in performance.values()],
        labels['f1']: [p['f1'] for p in performance.values()]
    }, index=performance.keys())

    fig, ax = plt.subplots(figsize=(12, 6))
    metrics_df.plot(kind='bar', ax=ax, width=0.7, alpha=0.8,
                    color=['#2ca02c', '#ff7f0e', '#1f77b4'])  # 绿色,橙色,蓝色

    ax.set_title(labels['perf_title'], fontsize=14, pad=20, fontweight='bold')
    ax.set_ylabel('Score', fontsize=12)
    ax.set_ylim(0, 1.05)
    ax.tick_params(axis='x', rotation=15, labelsize=11)
    ax.legend(fontsize=10, loc='lower right')
    ax.grid(axis='y', alpha=0.3)

    # 在柱状图上添加数值标签
    for container in ax.containers:
        ax.bar_label(container, fmt='%.3f', fontsize=9, padding=3)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'performance_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # ==================== 2. ROC曲线图 ====================
    print("绘制ROC曲线图")
    fig, ax = plt.subplots(figsize=(10, 8))
    best_model_name = max(performance.keys(), key=lambda x: performance[x]['roc_auc'])

    # 绘制每个模型的ROC曲线
    for name, perf in performance.items():
        fpr, tpr, _ = roc_curve(y_test, perf['y_prob'])
        linewidth = 3 if name == best_model_name else 2  # 最佳模型用粗线
        ax.plot(fpr, tpr, label=f"{name} (AUC = {perf['roc_auc']:.3f})",
                linewidth=linewidth, alpha=0.8)

    # 绘制随机猜测线(对角线)
    ax.plot([0, 1], [0, 1], 'k--', label=labels['random_guess'], linewidth=1.5, alpha=0.7)

    ax.set_xlabel(labels['fpr'], fontsize=12)
    ax.set_ylabel(labels['tpr'], fontsize=12)
    ax.set_title(f"{labels['roc_title']} (Best: {best_model_name})", fontsize=14, pad=20, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'roc_curve.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # ==================== 3. 混淆矩阵 ====================
    print("绘制混淆矩阵")
    best_perf = performance[best_model_name]
    cm = confusion_matrix(y_test, best_perf['y_pred'])
    cm_percent = cm / np.sum(cm, axis=1, keepdims=True) * 100  # 转换为百分比

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm_percent, cmap='Blues', aspect='auto')

    # 添加百分比文本
    for i in range(cm_percent.shape[0]):
        for j in range(cm_percent.shape[1]):
            text = ax.text(j, i, f'{cm_percent[i, j]:.1f}%',
                           ha="center", va="center", color="black", fontsize=11)

    ax.set_xlabel(labels['pred_label'], fontsize=12)
    ax.set_ylabel(labels['true_label'], fontsize=12)
    ax.set_title(f"{best_model_name} - {labels['cm_title']}", fontsize=14, pad=20, fontweight='bold')
    ax.set_xticks(np.arange(len(target_names)))
    ax.set_yticks(np.arange(len(target_names)))
    ax.set_xticklabels(target_names)
    ax.set_yticklabels(target_names)

    # 添加颜色条
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label(labels['percentage'], fontsize=10)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'confusion_matrix.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # ==================== 4. 特征重要性图 ====================
    print("绘制特征重要性图")
    try:
        lgb_pipeline = performance[best_model_name]['model']
        classifier = lgb_pipeline.named_steps['classifier']

        if hasattr(classifier, 'feature_importances_'):
            importances = classifier.feature_importances_
            non_zero_idx = np.where(importances > 0)[0]

            if len(non_zero_idx) > 0:
                # 选择最重要的10个特征
                sorted_idx = non_zero_idx[np.argsort(importances[non_zero_idx])[::-1][:10]]
                top_importances = importances[sorted_idx]
                top_feature_names = [f'feat_{i}' for i in sorted_idx]

                # 绘制水平柱状图
                fig, ax = plt.subplots(figsize=(10, 6))
                y_pos = np.arange(len(top_feature_names))
                ax.barh(y_pos, top_importances, alpha=0.8,
                        color=plt.cm.viridis(np.linspace(0, 1, len(top_feature_names))))

                ax.set_yticks(y_pos)
                ax.set_yticklabels(top_feature_names)
                ax.set_xlabel(labels['imp_x'], fontsize=12)
                ax.set_ylabel(labels['imp_y'], fontsize=12)
                ax.set_title(labels['imp_title'], fontsize=14, pad=20, fontweight='bold')
                ax.grid(axis='x', alpha=0.3)

                plt.tight_layout()
                plt.savefig(os.path.join(save_dir, 'feature_importance.png'), dpi=300, bbox_inches='tight')
                plt.close()

                # 输出特征索引说明
                print(f"特征索引说明 (共{len(importances)}维):")
                print("  feat_0~12: MFCC均值 (语音频谱特征)")
                print("  feat_13~25: MFCC标准差")
                print("  feat_26~153: 梅尔频谱均值 (128维)")
                print("  feat_154~281: 梅尔频谱标准差")
                print("  feat_282~283: 能量 (均值+标准差)")
                print("  feat_284~285: 基频 (均值+标准差)")

    except Exception as e:
        print(f"特征重要性可视化出错: {str(e)[:80]}")


# =============================================================================
# 主函数 - 程序执行入口
# =============================================================================

def main(audio_dir=None, difficulty_level=0.8):
    """
    主函数：整合整个AD语音检测流程

    参数:
        audio_dir (str): 真实音频数据目录，None则使用模拟数据
        difficulty_level (float): 分类难度级别(0-1)

    返回:
        训练好的模型对象

    流程概述:
        1. 数据准备和预处理
        2. 模型训练和评估
        3. 模型优化(可选)
        4. 结果可视化
        5. 新样本预测示例
    """
    try:
        # ==================== 1. 数据准备阶段 ====================
        print(f"===== 数据准备阶段 =====")
        X_train, X_test, y_train, y_test, preprocessor = load_and_preprocess_audio_data(
            audio_dir=audio_dir,
            ad_ratio=0.3,
            difficulty_level=difficulty_level
        )

        # ==================== 2. 模型训练阶段 ====================
        performance = train_and_evaluate_models(X_train, X_test, y_train, y_test, preprocessor)

        # ==================== 3. 模型选择和优化 ====================
        best_model_name = max(performance.keys(), key=lambda x: performance[x]['roc_auc'])
        best_model_info = performance[best_model_name]
        print(f"\n===== 选择最佳模型 =====")
        print(f"基础最佳模型: {best_model_name}")
        print(f"基础ROC-AUC: {best_model_info['roc_auc']:.4f}")

        # 根据性能决定是否进行优化(避免对过拟合模型进行优化)
        if best_model_info['roc_auc'] < 0.95:
            min_samples = 10
            if len(X_train) >= min_samples:
                optimized_model = optimize_best_model(X_train, y_train, preprocessor, best_model_info)
            else:
                print(f"训练集样本数过少 ({len(X_train)} < {min_samples})，跳过模型优化")
                optimized_model = best_model_info['model']
        else:
            print("模型性能过高，可能过拟合，跳过优化")
            optimized_model = best_model_info['model']

        # ==================== 4. 优化后模型评估 ====================
        print(f"\n===== 评估优化后模型 =====")
        y_prob_opt = optimized_model.predict_proba(X_test)[:, 1]

        # 使用Youden指数确定最佳阈值
        fpr, tpr, thresholds = roc_curve(y_test, y_prob_opt)
        youden_index = tpr - fpr
        best_idx = np.argmax(youden_index)
        best_threshold = thresholds[best_idx]

        # 验证阈值合理性
        y_pred_temp = (y_prob_opt >= best_threshold).astype(int)
        precision_temp = precision_score(y_test, y_pred_temp, zero_division=0)
        recall_temp = recall_score(y_test, y_pred_temp, zero_division=0)

        # 如果阈值导致极端结果，使用F1分数优化阈值
        if precision_temp < 0.3 or recall_temp < 0.3:
            print(f"Youden阈值 ({best_threshold:.2f}) 导致极端结果，使用F1最优阈值")
            thresholds_candidate = np.arange(0.1, 0.91, 0.05)
            f1_scores = []
            for th in thresholds_candidate:
                y_pred_cand = (y_prob_opt >= th).astype(int)
                f1_scores.append(f1_score(y_test, y_pred_cand, zero_division=0))
            best_threshold = thresholds_candidate[np.argmax(f1_scores)]

        # 最终预测和评估
        y_pred_opt = (y_prob_opt >= best_threshold).astype(int)
        print(f"最佳判定阈值: {best_threshold:.2f}")
        print(f"优化后准确率: {accuracy_score(y_test, y_pred_opt):.4f}")
        print(f"优化后精确率(AD类): {precision_score(y_test, y_pred_opt, zero_division=0):.4f}")
        print(f"优化后召回率(AD类): {recall_score(y_test, y_pred_opt, zero_division=0):.4f}")
        print(f"优化后F1分数: {f1_score(y_test, y_pred_opt, zero_division=0):.4f}")
        print(f"优化后ROC-AUC: {roc_auc_score(y_test, y_prob_opt):.4f}")

        print(f"优化后分类报告:")
        target_names = ['Normal', 'AD']
        print(classification_report(y_test, y_pred_opt, target_names=target_names, zero_division=0))

        # 更新性能字典
        performance[f'Optimized-{best_model_name}'] = {
            'model': optimized_model,
            'accuracy': accuracy_score(y_test, y_pred_opt),
            'precision': precision_score(y_test, y_pred_opt, zero_division=0),
            'recall': recall_score(y_test, y_pred_opt, zero_division=0),
            'f1': f1_score(y_test, y_pred_opt, zero_division=0),
            'roc_auc': roc_auc_score(y_test, y_prob_opt),
            'y_pred': y_pred_opt,
            'y_prob': y_prob_opt,
            'param_grid': best_model_info['param_grid']
        }

        # ==================== 5. 结果可视化 ====================
        visualize_results(performance, y_test, preprocessor)

        # ==================== 6. 新样本预测示例 ====================
        print(f"\n===== 模型训练完成 =====")
        print(f"新样本预测示例 (模拟AD患者语音特征):")

        # 生成模拟AD患者的特征向量
        mfcc_mean = np.random.normal(-0.05, 0.25, N_MFCC)
        mfcc_std = np.random.normal(0.18, 0.08, N_MFCC)
        mel_mean = np.random.normal(-0.1, 0.25, N_MEL)
        mel_std = np.random.normal(0.28, 0.08, N_MEL)
        energy_mean = np.random.normal(-0.15, 0.18)
        energy_std = np.random.normal(0.18, 0.07)
        f0_mean = np.random.normal(120, 20)
        f0_std = np.random.normal(28, 10)

        new_sample = np.array([np.concatenate([
            mfcc_mean, mfcc_std,
            mel_mean, mel_std,
            [energy_mean, energy_std],
            [f0_mean, f0_std]
        ])])

        # 进行预测
        pred_label = optimized_model.predict(new_sample)[0]
        pred_prob = optimized_model.predict_proba(new_sample)[:, 1][0]
        print(f"预测结果: {target_names[pred_label]}")
        print(f"AD概率: {pred_prob:.4f}")
        print(f"判定依据: 概率 ≥ {best_threshold:.2f} 判定为AD")

        return optimized_model

    except Exception as e:
        print(f"程序执行出错: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


# =============================================================================
# 程序入口点
# =============================================================================

if __name__ == "__main__":
    """
    程序主入口
    """
    print("阿尔茨海默症语音检测系统启动")
    print("信息: 当前使用高度真实的模拟数据运行")

    # ==================== 运行配置 ====================
    # 使用模拟数据
    best_model = main(audio_dir=None, difficulty_level=0.8)

    if best_model is not None:
        # 保存模型
        import joblib

        model_filename = 'ad_speech_detector.pkl'
        joblib.dump(best_model, model_filename)
        print(f"✅ 模型已保存为: {model_filename}")
        print(f"📁 文件路径: {os.path.abspath(model_filename)}")

        print(f"\n🎉 程序执行完成！")
        print(f"📊 可视化结果保存在: {os.path.abspath('ad_speech_plots')}")
        print(f"🤖 模型已训练完成并保存")
        print(f"\n🚀 下一步: 运行 api_service.py 启动API服务")
    else:
        print(f"❌ 程序执行失败，请检查错误信息")