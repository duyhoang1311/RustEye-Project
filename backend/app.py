from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import cv2
import numpy as np
import os
import shutil
import datetime
import yaml  # Thêm thư viện yaml để đọc tệp YAML
from werkzeug.utils import secure_filename  # Nhập secure_filename
from predictior import detect_and_segment_objects, predict_defect_details, process_video, process_and_predict_from_video_output, get_video_info
from ultralytics import YOLO

app = Flask(__name__)
CORS(app)

app.config['UPLOAD_FOLDER'] = "uploads"
app.config['PREDICTED_FOLDER'] = "predicted_images"

# Đường dẫn đến tệp weights và data.yaml
weights_path = "models/YOLOv9e-seg/best.pt"
data_yaml_path = "models/YOLOv9e-seg/data.yaml"

# Kiểm tra nếu weights_path tồn tại
if not os.path.isfile(weights_path):
    raise FileNotFoundError(f"Không tìm thấy tệp weights: {weights_path}")

# Kiểm tra nếu data_yaml_path tồn tại
if not os.path.isfile(data_yaml_path):
    raise FileNotFoundError(f"Không tìm thấy tệp data.yaml: {data_yaml_path}")

# Khởi tạo mô hình YOLO
yolo_model = YOLO(weights_path)

# Đọc dữ liệu từ tệp data.yaml (nếu cần thiết)
with open(data_yaml_path, 'r') as file:
    data_config = yaml.safe_load(file)

@app.route('/', methods=['GET'])
def home():
    return jsonify({'message': 'Backend is connected successfully!'}), 200

@app.route('/predicted_images/<path:filename>', methods=['GET'])
def serve_image(filename):
    try:
        # Xây dựng đường dẫn đầy đủ mà không có dấu ngoặc kép
        full_path = os.path.join(app.config['PREDICTED_FOLDER'], filename)
        print(f"Đường dẫn đầy đủ: {full_path}")  # In ra đường dẫn đầy đủ
        
        # Kiểm tra xem tệp có tồn tại không
        if not os.path.isfile(full_path):
            return jsonify({"error": "Tệp không tồn tại."}), 404
        return send_from_directory(app.config['PREDICTED_FOLDER'], filename)
    except Exception as e:
        print(f"Lỗi khi lấy file: {e}")  # In ra thông tin lỗi
        return jsonify({"error": "Không thể lấy file"}), 500

@app.route('/upload', methods=['POST'])
def upload_file():
    try:
        if 'image' not in request.files:
            return jsonify({"error": "Không tìm thấy file"}), 400

        file = request.files['image']
        if file.filename == '':
            return jsonify({"error": "Tên file trống"}), 400

        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        return jsonify({"message": f"File {filename} đã được upload thành công!"}), 200

    except Exception as e:
        return jsonify({"error": "Internal Server Error"}), 500

@app.route("/predict", methods=["POST"])
def predict():
    try:
        if 'image' not in request.files:
            return jsonify({"error": "Không có ảnh được tải lên"}), 400

        file = request.files['image']
        npimg = np.frombuffer(file.read(), np.uint8)
        image = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

        # Lưu tên file gốc
        original_filename = file.filename

        # Xóa các tệp đã dự đoán trước đó
        if os.path.exists(app.config['PREDICTED_FOLDER']):
            for filename in os.listdir(app.config['PREDICTED_FOLDER']):
                file_path = os.path.join(app.config['PREDICTED_FOLDER'], filename)
                try:
                    if os.path.isfile(file_path) or os.path.islink(file_path):
                        os.unlink(file_path)
                    elif os.path.isdir(file_path):
                        shutil.rmtree(file_path)
                except Exception as e:
                    print(f"Lỗi khi xóa file {file_path}: {e}")

        # Phát hiện và phân đoạn các đối tượng
        detected_objects = detect_and_segment_objects(image, weights_path)

        if detected_objects is None or not isinstance(detected_objects, list):
            return jsonify({"error": "Phát hiện không hợp lệ, vui lòng kiểm tra mô hình."}), 500

        results = predict_defect_details(image, detected_objects)

        # Lưu ảnh đã cắt và dự đoán vào thư mục predicted_images
        for i, obj in enumerate(detected_objects):
            x_min, y_min = obj['x_min'], obj['y_min']
            x_max, y_max = obj['x_max'], obj['y_max']
            mask = obj['mask']  # Mặt nạ phân đoạn (nhị phân)

            # Cắt ảnh theo vùng bounding box
            cropped_image = image[y_min:y_max, x_min:x_max]
            mask_cropped = mask[y_min:y_max, x_min:x_max]

            # Làm mờ nhẹ mặt nạ để giảm răng cưa
            mask_cropped_blurred = cv2.GaussianBlur(mask_cropped.astype(np.float32), (5, 5), sigmaX=1).astype(np.uint8)

            # Tạo ảnh với kênh alpha (RGBA) cho vùng đã cắt
            segmented_image_rgba = np.zeros((cropped_image.shape[0], cropped_image.shape[1], 4), dtype=np.uint8)

            # Áp dụng mặt nạ lên ảnh cắt để giữ lại phần đối tượng
            segmented_image_rgba[mask_cropped_blurred > 0.5, :3] = cropped_image[mask_cropped_blurred > 0.5]  # Kênh màu (RGB)
            segmented_image_rgba[mask_cropped_blurred > 0.5, 3] = 255  # Kênh alpha cho vùng đối tượng

            # Đặt pixel nền với alpha = 0 (trong suốt)
            segmented_image_rgba[mask_cropped_blurred <= 0.5, 3] = 0

            # Tạo tên file cho ảnh phân đoạn
            base_filename = os.path.splitext(original_filename)[0]
            segmented_image_filename = f'{base_filename}_predicted_{i}.png'

            # Lưu ảnh phân đoạn với nền trong suốt
            segmented_image_path = os.path.join(app.config['PREDICTED_FOLDER'], segmented_image_filename)
            cv2.imwrite(segmented_image_path, segmented_image_rgba)

        return jsonify({"results": results}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500
    
@app.route("/upload-video", methods=["POST"])
def upload_video():
    if "file" not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400

    filename = secure_filename(file.filename)
    file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    file.save(file_path)

    # Process video with OpenCV
    video_info = get_video_info(file_path)
    return jsonify(video_info)

@app.route("/predict_video", methods=["POST"])
def predict_video():
    try:
        if 'video' not in request.files:
            return jsonify({"error": "Không có video được tải lên"}), 400

        file = request.files['video']
        filename = secure_filename(file.filename)
        video_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(video_path)

        # Nhận giá trị frame_step từ request (mặc định là 500)
        frame_skip = int(request.form.get('frame_step', 500))

        # Xóa nội dung thư mục PREDICTED_FOLDER nếu đã tồn tại
        if os.path.exists(app.config['PREDICTED_FOLDER']):
            try:
                shutil.rmtree(app.config['PREDICTED_FOLDER'])
            except Exception as e:
                return jsonify({"error": f"Lỗi khi xóa thư mục: {str(e)}"}), 500

        os.makedirs(app.config['PREDICTED_FOLDER'], exist_ok=True)
        video_output_folder = os.path.join(app.config['PREDICTED_FOLDER'], f"{filename}_frames")
        os.makedirs(video_output_folder, exist_ok=True)

        # Xử lý video
        results = process_video(video_path, video_output_folder, weights_path, frame_skip)
        server_url = request.host_url.rstrip('/')
        all_results, summary = process_and_predict_from_video_output(video_output_folder, weights_path, server_url)

        return jsonify({
            "message": "Dự đoán trên video hoàn tất!",
            "results": all_results,
            "summary": summary
        }), 200

    except Exception as e:
        # Đảm bảo luôn trả về JSON hợp lệ khi có lỗi
        return jsonify({"error": f"Lỗi xảy ra: {str(e)}"}), 500



if __name__ == "__main__":
    app.run(debug=True)