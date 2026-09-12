"""
下载离线vendor资源（Bootstrap CSS/JS、Bootstrap Icons、Chart.js）
用于本地优先+CDN备用的资源加载策略
"""
import os
import urllib.request
import ssl

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
VENDOR_DIR = os.path.join(BASE_DIR, 'static', 'vendor')

RESOURCES = {
    'css/bootstrap.min.css': 'https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/css/bootstrap.min.css',
    'css/bootstrap-icons.min.css': 'https://cdn.jsdelivr.net/npm/bootstrap-icons@1.11.2/font/bootstrap-icons.min.css',
    'js/bootstrap.bundle.min.js': 'https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/js/bootstrap.bundle.min.js',
    'js/chart.min.js': 'https://cdn.jsdelivr.net/npm/chart.js@4.4.1/dist/chart.umd.min.js',
}

# Bootstrap Icons字体文件
FONT_RESOURCES = {
    'fonts/bootstrap-icons.woff2': 'https://cdn.jsdelivr.net/npm/bootstrap-icons@1.11.2/font/fonts/bootstrap-icons.woff2',
    'fonts/bootstrap-icons.woff': 'https://cdn.jsdelivr.net/npm/bootstrap-icons@1.11.2/font/fonts/bootstrap-icons.woff',
}


def download_file(url, filepath):
    """下载文件"""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    if os.path.exists(filepath):
        print(f"  [跳过] 已存在: {os.path.basename(filepath)}")
        return True
    try:
        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, context=ctx, timeout=30) as response:
            data = response.read()
        with open(filepath, 'wb') as f:
            f.write(data)
        size_kb = len(data) / 1024
        print(f"  [下载] {os.path.basename(filepath)} ({size_kb:.1f} KB)")
        return True
    except Exception as e:
        print(f"  [失败] {os.path.basename(filepath)}: {e}")
        return False


def fix_icon_css_font_path(css_path):
    """修正Bootstrap Icons CSS中的字体路径为本地路径"""
    if not os.path.exists(css_path):
        return
    with open(css_path, 'r', encoding='utf-8') as f:
        content = f.read()
    # 将字体url路径替换为本地相对路径
    content = content.replace('url("./fonts/', 'url("../fonts/')
    with open(css_path, 'w', encoding='utf-8') as f:
        f.write(content)
    print("  [修正] bootstrap-icons.min.css 字体路径已更新")


def main():
    print("=" * 50)
    print("下载离线Vendor资源")
    print("=" * 50)

    print("\n下载CSS/JS资源:")
    for rel_path, url in RESOURCES.items():
        filepath = os.path.join(VENDOR_DIR, rel_path)
        download_file(url, filepath)

    print("\n下载字体资源:")
    for rel_path, url in FONT_RESOURCES.items():
        filepath = os.path.join(VENDOR_DIR, rel_path)
        download_file(url, filepath)

    # 修正CSS中字体路径
    icon_css = os.path.join(VENDOR_DIR, 'css', 'bootstrap-icons.min.css')
    fix_icon_css_font_path(icon_css)

    print("\n" + "=" * 50)
    print("离线资源下载完成!")
    print(f"资源目录: {VENDOR_DIR}")
    print("=" * 50)


if __name__ == '__main__':
    main()
