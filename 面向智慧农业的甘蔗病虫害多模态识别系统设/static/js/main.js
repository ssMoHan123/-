/**
 * 甘蔗病虫害多模态识别系统 - 公共JavaScript
 */

// 页面加载完成后执行
document.addEventListener('DOMContentLoaded', function() {
    console.log('甘蔗病虫害多模态识别系统已加载');

    // 自动关闭提示消息
    const alerts = document.querySelectorAll('.alert-dismissible');
    alerts.forEach(function(alert) {
        setTimeout(function() {
            var bsAlert = bootstrap.Alert.getOrCreateInstance(alert);
            if (bsAlert) bsAlert.close();
        }, 5000);
    });
});

/**
 * 格式化文件大小
 */
function formatFileSize(bytes) {
    if (bytes === 0) return '0 Bytes';
    var k = 1024;
    var sizes = ['Bytes', 'KB', 'MB', 'GB'];
    var i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
}

/**
 * 格式化日期
 */
function formatDate(dateStr) {
    var date = new Date(dateStr);
    return date.toLocaleString('zh-CN');
}