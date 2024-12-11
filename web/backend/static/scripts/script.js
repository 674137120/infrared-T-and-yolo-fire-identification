// 后端API基础URL，根据您的后端部署地址修改
const BASE_URL = 'http://192.168.175.72:5000';

// 更新图片检测置信度显示
function updateConfValue(value) {
    const conf = (value / 100).toFixed(2);
    document.getElementById('confValue').innerText = conf;
}

// 更新视频检测置信度显示
function updateVidConfValue(value) {
    const conf = (value / 100).toFixed(2);
    document.getElementById('vidConfValue').innerText = conf;
}

// 更新温度阈值显示
function updateTempValue(value) {
    document.getElementById('tempValue').innerText = `${value}°C`;
}

// 上传图片并检测
async function uploadImage() {
    const fileInput = document.getElementById('imageUpload');
    const file = fileInput.files[0];
    if (!file) {
        alert("请先选择一张图片。");
        return;
    }

    const confThreshold = document.getElementById('confSlider').value / 100;
    // 这里假设没有单独的图片温度阈值调整控件，如需可自行添加
    const tempThreshold = 800; 

    document.getElementById('messageImage').style.display = 'block';
    document.getElementById('detectedImage').style.display = 'none';

    const formData = new FormData();
    formData.append('image', file);
    formData.append('conf_thres', confThreshold);
    formData.append('temp_thres', tempThreshold);

    try {
        const response = await fetch(`${BASE_URL}/api/detect_image`, {
            method: 'POST',
            body: formData
        });

        if (!response.ok) {
            const errorData = await response.json();
            alert(`图像检测失败: ${errorData.error}`);
            document.getElementById('messageImage').style.display = 'none';
            return;
        }

        const data = await response.json();
        const detectedImage = document.getElementById('detectedImage');
        detectedImage.src = `${BASE_URL}${data.detected_image_url}`;
        detectedImage.style.display = 'block';
        document.getElementById('messageImage').style.display = 'none';

    } catch (e) {
        alert(`请求失败: ${e}`);
        document.getElementById('messageImage').style.display = 'none';
    }
}

// 保存检测结果图片
function saveImage() {
    const img = document.getElementById('detectedImage').src;
    if (!img) {
        alert("没有检测到图像。");
        return;
    }
    const link = document.createElement('a');
    link.href = img;
    link.download = 'detected_image.jpg';
    link.click();
}

// 上传视频并检测
async function uploadVideo() {
    const fileInput = document.getElementById('videoUpload');
    const file = fileInput.files[0];
    if (!file) {
        alert("请先选择一个视频文件。");
        return;
    }

    const confThreshold = document.getElementById('vidConfSlider').value / 100;
    const tempThreshold = document.getElementById('tempSlider').value;

    document.getElementById('messageVideo').style.display = 'block';
    document.getElementById('detectedVideo').style.display = 'none';

    const formData = new FormData();
    formData.append('video', file);
    formData.append('conf_thres', confThreshold);
    formData.append('temp_thres', tempThreshold);

    try {
        const response = await fetch(`${BASE_URL}/api/detect_video`, {
            method: 'POST',
            body: formData
        });

        if (!response.ok) {
            const errorData = await response.json();
            alert(`视频检测失败: ${errorData.error}`);
            document.getElementById('messageVideo').style.display = 'none';
            return;
        }

        const data = await response.json();
        const detectedVideo = document.getElementById('detectedVideo');
        detectedVideo.src = `${BASE_URL}${data.detected_video_url}`;
        detectedVideo.style.display = 'block';
        document.getElementById('messageVideo').style.display = 'none';

    } catch (e) {
        alert(`请求失败: ${e}`);
        document.getElementById('messageVideo').style.display = 'none';
    }
}

// 停止检测请求（需后端支持）
async function stopDetection() {
    try {
        const response = await fetch(`${BASE_URL}/api/stop_detection`, {
            method: 'POST'
        });

        if (response.ok) {
            alert("检测已停止。");
            // 如有需要可清空video显示等操作
        } else {
            const errorData = await response.json();
            alert(`无法停止检测: ${errorData.error}`);
        }
    } catch (e) {
        alert(`请求失败: ${e}`);
    }
}
