"""
数据预处理脚本：将YOLO格式的检测数据集转换为分类数据集
通过读取每张图片对应的YOLO标签，取主要类别作为图片的分类标签
"""
import os
import shutil
from collections import Counter
from config import (DATA_DIR, TRAIN_DIR, VALID_DIR, TEST_DIR,
                    CLASSIFIED_DIR, CLASS_NAMES)


def get_dominant_class(label_path):
    """从YOLO标签文件获取主要类别ID"""
    if not os.path.exists(label_path):
        return None
    with open(label_path, 'r') as f:
        lines = f.readlines()
    if not lines:
        return None
    class_ids = []
    for line in lines:
        parts = line.strip().split()
        if parts:
            class_ids.append(int(parts[0]))
    if not class_ids:
        return None
    # 返回出现最多的类别
    counter = Counter(class_ids)
    return counter.most_common(1)[0][0]


def process_split(split_dir, output_base, split_name):
    """处理一个数据集分割（train/valid/test）"""
    images_dir = os.path.join(split_dir, 'images')
    labels_dir = os.path.join(split_dir, 'labels')

    if not os.path.exists(images_dir):
        print(f"[警告] 图片目录不存在: {images_dir}")
        return

    stats = Counter()
    skipped = 0

    for img_name in os.listdir(images_dir):
        if not img_name.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
            continue

        # 获取对应的标签文件
        label_name = os.path.splitext(img_name)[0] + '.txt'
        label_path = os.path.join(labels_dir, label_name)

        class_id = get_dominant_class(label_path)
        if class_id is None or class_id >= len(CLASS_NAMES):
            skipped += 1
            continue

        class_name = CLASS_NAMES[class_id]
        # 创建类别目录
        class_dir = os.path.join(output_base, split_name, class_name)
        os.makedirs(class_dir, exist_ok=True)

        # 复制图片
        src = os.path.join(images_dir, img_name)
        dst = os.path.join(class_dir, img_name)
        if not os.path.exists(dst):
            shutil.copy2(src, dst)

        stats[class_name] += 1

    print(f"\n[{split_name}] 处理完成:")
    for cls_name in CLASS_NAMES:
        print(f"  {cls_name}: {stats.get(cls_name, 0)} 张")
    print(f"  跳过: {skipped} 张")
    print(f"  总计: {sum(stats.values())} 张")


def main():
    print("=" * 60)
    print("甘蔗病虫害数据集预处理：YOLO格式 -> 分类格式")
    print("=" * 60)

    # 清理旧数据
    if os.path.exists(CLASSIFIED_DIR):
        print(f"\n清理旧的分类数据目录: {CLASSIFIED_DIR}")
        shutil.rmtree(CLASSIFIED_DIR)

    os.makedirs(CLASSIFIED_DIR, exist_ok=True)

    # 处理各个分割
    process_split(TRAIN_DIR, CLASSIFIED_DIR, 'train')
    process_split(VALID_DIR, CLASSIFIED_DIR, 'valid')
    process_split(TEST_DIR, CLASSIFIED_DIR, 'test')

    print("\n" + "=" * 60)
    print("数据预处理完成！")
    print(f"分类数据保存在: {CLASSIFIED_DIR}")
    print("=" * 60)


if __name__ == '__main__':
    main()
