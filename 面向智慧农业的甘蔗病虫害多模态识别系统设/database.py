"""
SQLite 数据库模块
管理识别记录、环境数据等
"""
import os
import sqlite3
from datetime import datetime
from config import DATABASE_PATH


def get_db():
    """获取数据库连接"""
    os.makedirs(os.path.dirname(DATABASE_PATH), exist_ok=True)
    conn = sqlite3.connect(DATABASE_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


def init_db():
    """初始化数据库表"""
    conn = get_db()
    cursor = conn.cursor()

    # 识别记录表
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS recognition_records (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            image_filename TEXT NOT NULL,
            image_path TEXT NOT NULL,
            predicted_class TEXT NOT NULL,
            predicted_class_cn TEXT NOT NULL,
            confidence REAL NOT NULL,
            treatment_advice TEXT,
            temperature REAL,
            humidity REAL,
            rainfall REAL,
            soil_moisture REAL,
            notes TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')

    # 模型信息表
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS model_info (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            model_name TEXT NOT NULL,
            model_path TEXT NOT NULL,
            accuracy REAL,
            total_epochs INTEGER,
            classes TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')

    # 环境数据表
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS environment_data (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            record_id INTEGER,
            temperature REAL,
            humidity REAL,
            rainfall REAL,
            soil_moisture REAL,
            wind_speed REAL,
            light_intensity REAL,
            recorded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (record_id) REFERENCES recognition_records(id)
        )
    ''')

    conn.commit()
    conn.close()
    print("数据库初始化完成")


def insert_recognition_record(image_filename, image_path, predicted_class,
                               predicted_class_cn, confidence, treatment_advice,
                               temperature=None, humidity=None, rainfall=None,
                               soil_moisture=None, notes=None):
    """插入识别记录"""
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO recognition_records
        (image_filename, image_path, predicted_class, predicted_class_cn,
         confidence, treatment_advice, temperature, humidity, rainfall,
         soil_moisture, notes, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (image_filename, image_path, predicted_class, predicted_class_cn,
          confidence, treatment_advice, temperature, humidity, rainfall,
          soil_moisture, notes, datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
    record_id = cursor.lastrowid

    # 如果有环境数据，也插入环境数据表
    if any(v is not None for v in [temperature, humidity, rainfall, soil_moisture]):
        cursor.execute('''
            INSERT INTO environment_data
            (record_id, temperature, humidity, rainfall, soil_moisture, recorded_at)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (record_id, temperature, humidity, rainfall, soil_moisture,
              datetime.now().strftime('%Y-%m-%d %H:%M:%S')))

    conn.commit()
    conn.close()
    return record_id


def get_all_records(page=1, per_page=20, search=None, class_filter=None):
    """获取所有识别记录（分页）"""
    conn = get_db()
    cursor = conn.cursor()

    query = 'SELECT * FROM recognition_records WHERE 1=1'
    params = []

    if search:
        query += ' AND (image_filename LIKE ? OR predicted_class_cn LIKE ? OR notes LIKE ?)'
        params.extend([f'%{search}%', f'%{search}%', f'%{search}%'])

    if class_filter and class_filter != 'all':
        query += ' AND predicted_class = ?'
        params.append(class_filter)

    # 获取总数
    count_query = query.replace('SELECT *', 'SELECT COUNT(*)')
    cursor.execute(count_query, params)
    total = cursor.fetchone()[0]

    # 分页查询
    query += ' ORDER BY created_at DESC LIMIT ? OFFSET ?'
    params.extend([per_page, (page - 1) * per_page])
    cursor.execute(query, params)
    records = [dict(row) for row in cursor.fetchall()]

    conn.close()
    return records, total


def get_record_by_id(record_id):
    """根据ID获取记录"""
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM recognition_records WHERE id = ?', (record_id,))
    record = cursor.fetchone()
    conn.close()
    return dict(record) if record else None


def get_statistics():
    """获取统计数据"""
    conn = get_db()
    cursor = conn.cursor()

    stats = {}

    # 总记录数
    cursor.execute('SELECT COUNT(*) FROM recognition_records')
    stats['total_records'] = cursor.fetchone()[0]

    # 各类别统计
    cursor.execute('''
        SELECT predicted_class, predicted_class_cn, COUNT(*) as count
        FROM recognition_records
        GROUP BY predicted_class
        ORDER BY count DESC
    ''')
    stats['class_distribution'] = [dict(row) for row in cursor.fetchall()]

    # 平均置信度
    cursor.execute('SELECT AVG(confidence) FROM recognition_records')
    avg_conf = cursor.fetchone()[0]
    stats['avg_confidence'] = round(avg_conf * 100, 2) if avg_conf else 0

    # 最近7天每天的记录数
    cursor.execute('''
        SELECT DATE(created_at) as date, COUNT(*) as count
        FROM recognition_records
        WHERE created_at >= DATE('now', '-7 days')
        GROUP BY DATE(created_at)
        ORDER BY date
    ''')
    stats['daily_records'] = [dict(row) for row in cursor.fetchall()]

    # 病害检出率（非healthy的比例）
    cursor.execute('''
        SELECT COUNT(*) FROM recognition_records WHERE predicted_class != 'healthy'
    ''')
    disease_count = cursor.fetchone()[0]
    stats['disease_rate'] = round(disease_count / max(stats['total_records'], 1) * 100, 2)

    conn.close()
    return stats


def delete_record(record_id):
    """删除记录"""
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute('DELETE FROM environment_data WHERE record_id = ?', (record_id,))
    cursor.execute('DELETE FROM recognition_records WHERE id = ?', (record_id,))
    conn.commit()
    conn.close()


def save_model_info(model_name, model_path, accuracy, total_epochs, classes):
    """保存模型信息"""
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO model_info (model_name, model_path, accuracy, total_epochs, classes)
        VALUES (?, ?, ?, ?, ?)
    ''', (model_name, model_path, accuracy, total_epochs, str(classes)))
    conn.commit()
    conn.close()


if __name__ == '__main__':
    init_db()
    print("数据库创建成功!")
