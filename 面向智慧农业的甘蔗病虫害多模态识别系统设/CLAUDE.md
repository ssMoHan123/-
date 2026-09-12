# CLAUDE.md

本文件为 Claude Code（claude.ai/code）在此仓库中工作时提供指导。

## 项目概述

基于 Flask + MobileNetV2 的深度学习甘蔗病虫害智能识别 Web 系统，面向智慧农业场景。支持图像/视频识别（6 种病害类型）、文本症状诊断、环境数据融合辅助诊断。前端采用离线优先 + CDN 备用的资源加载策略。

## 常用命令

```bash
# 安装依赖
pip install -r requirements.txt

# 下载离线前端资源（Bootstrap、Chart.js、Bootstrap Icons）
python download_vendor.py

# 数据预处理：将 YOLO 检测格式转为 ImageFolder 分类格式
python prepare_data.py

# 训练 MobileNetV2 模型（验证准确率达 85% 或连续 3 轮无改进时自动停止）
python train.py

# 启动 Web 应用（访问 http://127.0.0.1:5000）
python app.py
```

## 架构

**配置中心**：`config.py` 定义所有共享常量——路径、类别名称（中英文映射）、症状关键词字典、训练超参数、Flask 配置。所有其他模块均从此文件导入。

**数据管线**：`data/` 目录中的 YOLO 格式数据集 → `prepare_data.py` 读取 YOLO `.txt` 标签，取每张图片中出现最多的类别作为标签，将图片按 `classified_data/{train,valid,test}/{类别名}/` 的 PyTorch `ImageFolder` 结构组织 → `train.py` 冻结 MobileNetV2 主干网络，替换自定义分类头进行迁移学习 → 模型保存至 `models/best_mobilenet.pth`。

**模型架构**（`app.py:57-65`，`train.py:48-65`）：使用 ImageNet 预训练的 MobileNetV2。训练时冻结除最后 3 层外的所有特征层，分类头替换为 Dropout→Linear(256)→ReLU→Dropout→Linear(类别数)。推理时 `app.py` 加载保存的 checkpoint 并构建相同架构。

**Flask 应用**（`app.py`）：单体应用，所有路由、模型加载（启动时调用 `load_model()`）、图像/视频推理、文本匹配逻辑均在此。路由一览：
- `/` — 首页仪表盘，展示统计数据
- `/recognize` GET/POST — 图片上传 + 预测 + 可选环境数据
- `/recognize_video` POST — 视频帧提取（OpenCV，每 N 帧采样）+ 投票分类
- `/video` — 视频识别页面
- `/symptom` GET/POST — 基于关键词匹配的文本症状诊断
- `/history` — 分页历史记录，支持搜索和按类别筛选
- `/record/<id>` — 记录详情
- `/statistics` — 基于 Chart.js 的统计分析
- `/about` — 模型信息展示

**数据库**（`database.py`）：使用 `sqlite3` 操作 SQLite 数据库，开启 WAL 模式。三张表：`recognition_records`、`model_info`、`environment_data`。所有查询使用原生 SQL 参数化语句。

**前端**（`templates/base.html`）：Jinja2 基础模板，Bootstrap CSS/JS 采用本地优先加载 + CDN 自动容错回退机制。导航栏链接所有页面。`main.js` 提供共享工具函数（文件大小格式化、日期格式化、自动关闭提示框）。统计分析页使用 Chart.js 渲染图表。

**病害类别**：共 6 类 —— `cercospora`（褐斑病）、`eyespot`（眼斑病）、`healthy`（健康）、`redrot`（赤腐病）、`wheat_rust`（锈病）、`yellow_leaf`（黄叶病）。类别映射及防治建议均定义在 `config.py` 中。

**Windows 注意事项**：`config.py:82` 中 `num_workers` 设为 0，以避免 Windows 下多进程问题。未经 Windows 测试不要修改此值。

## 注意事项
用中文和我对话
每句话后面都要加一句“喵~”
例：关注塔菲，关注塔菲谢谢 -> 关注塔菲喵~关注塔菲谢谢喵~
