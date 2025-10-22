"""
创建不依赖自定义函数的简化模型
"""

import numpy as np
import joblib
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_simple_ad_model():
    """创建简化的AD检测模型"""
    logger.info("创建简化AD检测模型...")

    # 生成模拟数据
    n_samples = 1000
    n_features = 286

    # 创建特征数据
    np.random.seed(42)
    X = np.random.normal(0, 1, (n_samples, n_features))

    # 创建标签 - 基于特征的简单逻辑
    # 让前50个特征对AD有影响
    ad_weights = np.zeros(n_features)
    ad_weights[:50] = np.random.normal(0, 0.5, 50)

    # 计算AD概率
    linear_combination = X @ ad_weights
    ad_prob = 1 / (1 + np.exp(-linear_combination))
    y = (ad_prob > 0.5).astype(int)

    # 调整类别比例
    ad_ratio = 0.3
    n_ad = int(n_samples * ad_ratio)
    ad_indices = np.where(y == 1)[0]
    non_ad_indices = np.where(y == 0)[0]

    if len(ad_indices) > n_ad:
        # 随机选择部分AD样本转为非AD
        to_convert = np.random.choice(ad_indices, len(ad_indices) - n_ad, replace=False)
        y[to_convert] = 0
    elif len(ad_indices) < n_ad:
        # 随机选择部分非AD样本转为AD
        to_convert = np.random.choice(non_ad_indices, n_ad - len(ad_indices), replace=False)
        y[to_convert] = 1

    logger.info(f"数据分布 - AD: {np.sum(y)}, 正常: {len(y) - np.sum(y)}")

    # 创建简单的管道模型
    model = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', RandomForestClassifier(
            n_estimators=50,
            max_depth=5,
            random_state=42,
            class_weight='balanced'
        ))
    ])

    # 训练模型
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model.fit(X_train, y_train)

    # 评估模型
    train_score = model.score(X_train, y_train)
    test_score = model.score(X_test, y_test)

    logger.info(f"训练准确率: {train_score:.4f}")
    logger.info(f"测试准确率: {test_score:.4f}")

    return model


def main():
    """主函数"""
    logger.info("开始创建简化模型...")

    try:
        # 创建模型
        model = create_simple_ad_model()

        # 保存模型
        model_filename = 'ad_model_simple.pkl'
        joblib.dump(model, model_filename)

        logger.info(f"✅ 模型保存成功: {model_filename}")

        # 测试模型加载
        loaded_model = joblib.load(model_filename)
        logger.info("✅ 模型加载测试成功")

        # 测试预测
        test_sample = np.random.normal(0, 1, (1, 286))
        prediction = loaded_model.predict_proba(test_sample)
        logger.info(f"✅ 预测测试成功 - 概率: {prediction[0]}")

        print(f"\n🎉 简化模型创建成功！")
        print(f"📁 模型文件: {model_filename}")
        print(f"🔧 下一步: 修改 new.api.py 使用新模型文件")

    except Exception as e:
        logger.error(f"❌ 模型创建失败: {e}")
        raise


if __name__ == "__main__":
    main()