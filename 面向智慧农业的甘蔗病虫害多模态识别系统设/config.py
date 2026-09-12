import os

# 项目根目录
BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def _load_env():
    """加载 .env 文件中的环境变量（不覆盖已存在的系统环境变量）"""
    env_path = os.path.join(BASE_DIR, '.env')
    if not os.path.exists(env_path):
        return
    with open(env_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#') or '=' not in line:
                continue
            key, value = line.split('=', 1)
            os.environ.setdefault(key.strip(), value.strip())


_load_env()

# 数据目录
DATA_DIR = os.path.join(BASE_DIR, 'data')
TRAIN_DIR = os.path.join(DATA_DIR, 'train')
VALID_DIR = os.path.join(DATA_DIR, 'valid')
TEST_DIR = os.path.join(DATA_DIR, 'test')

# 分类数据目录（预处理后）
CLASSIFIED_DIR = os.path.join(BASE_DIR, 'classified_data')

# 模型保存目录
MODEL_DIR = os.path.join(BASE_DIR, 'models')
BEST_MODEL_PATH = os.path.join(MODEL_DIR, 'best_mobilenet.pth')

# 上传文件目录
UPLOAD_DIR = os.path.join(BASE_DIR, 'static', 'uploads')

# 数据库
DATABASE_PATH = os.path.join(BASE_DIR, 'database', 'sugarcane.db')

# 类别名称（中英文映射）
CLASS_NAMES = ['cercospora', 'eyespot', 'healthy', 'redrot', 'wheat_rust', 'yellow_leaf']
CLASS_NAMES_CN = {
    'cercospora': '褐斑病',
    'eyespot': '眼斑病',
    'healthy': '健康',
    'redrot': '赤腐病',
    'wheat_rust': '锈病',
    'yellow_leaf': '黄叶病',
}

# 病害防治建议
TREATMENT_ADVICE = {
    'cercospora': '建议使用代森锰锌或百菌清等杀菌剂喷施，注意排水降湿，合理密植。',
    'eyespot': '建议使用三唑类杀菌剂防治，及时清除病残体，增施钾肥增强抗病力。',
    'healthy': '植株健康，继续保持良好的田间管理，定期巡查。',
    'redrot': '建议选用抗病品种，种植前用多菌灵浸种，发现病株及时拔除烧毁。',
    'wheat_rust': '建议使用三唑酮或丙环唑等杀菌剂喷施，注意田间通风，合理施肥。',
    'yellow_leaf': '建议选用抗病品种，加强水肥管理，使用脱毒健康种苗，及时防治蚜虫。',
}

# 症状关键词映射（用于文本多模态识别）
SYMPTOM_KEYWORDS = {
    'cercospora': {
        'keywords': ['褐斑', '褐色斑点', '棕色斑', '叶斑', '尾孢', '褐色病斑', '圆形斑点', '叶片褐色'],
        'description': '褐斑病(Cercospora): 叶片出现褐色或棕色圆形/椭圆形斑点，由尾孢菌引起的真菌性病害。',
    },
    'eyespot': {
        'keywords': ['眼斑', '眼状', '椭圆斑', '灰白中央', '眼形', '梭形斑', '中央灰白', '边缘褐色斑'],
        'description': '眼斑病(Eyespot): 叶片出现椭圆形眼状斑点，中央灰白色，边缘褐色。',
    },
    'healthy': {
        'keywords': ['健康', '正常', '无病', '生长良好', '无症状', '无斑点', '叶片绿色', '长势好'],
        'description': '健康(Healthy): 甘蔗植株生长正常，叶片绿色，无明显病害症状。',
    },
    'redrot': {
        'keywords': ['赤腐', '红腐', '红色腐烂', '茎腐', '红斑', '茎部变红', '内部红色', '腐烂发红'],
        'description': '赤腐病(Red Rot): 甘蔗茎部和叶片出现红色腐烂症状，是一种严重的真菌性病害。',
    },
    'wheat_rust': {
        'keywords': ['锈病', '锈斑', '铁锈', '锈色', '粉状', '孢子堆', '橙色粉末', '黄褐色粉'],
        'description': '锈病(Wheat Rust): 叶片表面出现铁锈色粉状孢子堆，是一种真菌性病害。',
    },
    'yellow_leaf': {
        'keywords': ['黄叶', '叶片变黄', '发黄', '黄化', '中脉变黄', '叶脉黄', '叶片枯黄', '黄色条纹'],
        'description': '黄叶病(Yellow Leaf): 叶片从中脉开始逐渐变黄，是一种病毒性病害。',
    },
}

# 训练参数
TRAIN_CONFIG = {
    'image_size': 224,
    'batch_size': 32,
    'learning_rate': 0.001,
    'max_epochs': 50,
    'target_accuracy': 0.85,
    'patience': 3,
    'num_workers': 0,  # Windows下设为0避免多进程问题
}

# Flask配置
FLASK_CONFIG = {
    'SECRET_KEY': 'sugarcane-disease-recognition-2024',
    'MAX_CONTENT_LENGTH': 16 * 1024 * 1024,  # 16MB
}

# DeepSeek大模型API配置
# api_key 从环境变量读取，避免硬编码泄露；请在 .env 文件或系统环境变量中配置 DEEPSEEK_API_KEY
DEEPSEEK_CONFIG = {
    'api_key': os.environ.get('DEEPSEEK_API_KEY', ''),
    'api_url': 'https://api.deepseek.com/v1/chat/completions',
    'model': 'deepseek-chat',
    'max_tokens': 1024,
    'temperature': 0.7,
}
