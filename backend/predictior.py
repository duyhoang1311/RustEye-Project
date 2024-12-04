import cv2
import os
import numpy as np
from keras.models import load_model
from keras.applications.densenet import preprocess_input
import matplotlib.pyplot as plt
from skfuzzy import control as ctrl
import skfuzzy as fuzz
from ortools.sat.python import cp_model
from ultralytics import YOLO
import json
import math


# Load mô hình phân loại
model = load_model("models/DenseNet121/densenet121.h5")
class_names = ['Bong tróc sơn', 'Rỉ sét', "Vết nứt", "Dây cáp rỉ sét"]

detected_objects = []

def detect_and_segment_objects(image, model_path):
    model = YOLO(model_path)
    img_height, img_width = image.shape[:2]
    detected_objects = []  # Khởi tạo danh sách ở đây

    results = model(image, imgsz=416, conf=0.1)

    for result in results:
        masks = result.masks.data.cpu().numpy()
        classes = result.boxes.cls.cpu().numpy()
        confidences = result.boxes.conf.cpu().numpy()

        for i, mask in enumerate(masks):
            mask = np.squeeze(mask)
            mask_resized = cv2.resize(mask, (img_width, img_height))

            y, x = np.where(mask_resized == 1)
            if len(x) > 0 and len(y) > 0:
                min_x, max_x = np.min(x), np.max(x)
                min_y, max_y = np.min(y), np.max(y)

                class_id = int(classes[i])
                confidence = confidences[i]
                class_name = model.names[class_id]

                detected_objects.append({
                    'x_min': min_x,
                    'y_min': min_y,
                    'x_max': max_x,
                    'y_max': max_y,
                    'class_name': class_name,
                    'confidence': confidence,
                    'defect_type': None,
                    'img_width': img_width,  # Thêm chiều rộng gốc của ảnh
                    'img_height': img_height,  # Thêm chiều cao gốc của ảnh
                    'mask': mask_resized
                })

    print("Detected Objects:", detected_objects)  # In danh sách đối tượng đã phát hiện
    return detected_objects  # Trả về danh sách



def calculate_damage_level(mask, img_width, img_height):
    damage_area = np.sum(mask == 1)  
    total_area = img_width * img_height
    return (damage_area / total_area) * 100


def fuzzy_logic_damage_level(damage_percentage, predicted_class):
     # Define fuzzy variables for damage size 
    damage_size = ctrl.Antecedent(np.arange(0, 101, 1), 'damage_size')
    damage_level_fuzzy = ctrl.Consequent(np.arange(0, 101, 1), 'damage_level')

    # Membership functions for damage size based on ISO 8501-3 classes (Ri0, Ri1, Ri2, Ri3, Ri4, Ri5)
    damage_size['Ri0'] = fuzz.trimf(damage_size.universe, [0, 0, 0.05])            
    damage_size['Ri1'] = fuzz.trimf(damage_size.universe, [0, 0.05, 0.5])    
    damage_size['Ri2'] = fuzz.trimf(damage_size.universe, [0.05, 0.5, 1])   
    damage_size['Ri3'] = fuzz.trimf(damage_size.universe, [0.5, 1, 8])       
    damage_size['Ri4'] = fuzz.trimf(damage_size.universe, [1, 8, 40])          
    damage_size['Ri5'] = fuzz.trimf(damage_size.universe, [8, 40, 50])      
         
    # Membership functions for damage size based on ISO 8501-4 classes (VN0, VN1, VN2, VN3, VN4, VN5)
    damage_size['VN0'] = fuzz.trimf(damage_size.universe, [0, 0, 0.01])       
    damage_size['VN1'] = fuzz.trimf(damage_size.universe, [0.01, 0.05, 0.1]) 
    damage_size['VN2'] = fuzz.trimf(damage_size.universe, [0.1, 0.2, 0.2])    
    damage_size['VN3'] = fuzz.trimf(damage_size.universe, [0.2, 0.35, 0.5])   
    damage_size['VN4'] = fuzz.trimf(damage_size.universe, [0.5, 0.75, 1])     
    damage_size['VN5'] = fuzz.trimf(damage_size.universe, [1, 2, 10])         

    # Membership functions for damage size based on ISO 8501-5 classes (BT0, BT1, BT2, BT3, BT4, BT5)
    damage_size['BT0'] = fuzz.trimf(damage_size.universe, [0, 0, 0.1])       
    damage_size['BT1'] = fuzz.trimf(damage_size.universe, [0.1, 0.5, 1])     
    damage_size['BT2'] = fuzz.trimf(damage_size.universe, [1, 2, 3])         
    damage_size['BT3'] = fuzz.trimf(damage_size.universe, [3, 6, 10])        
    damage_size['BT4'] = fuzz.trimf(damage_size.universe, [10, 20, 30])      
    damage_size['BT5'] = fuzz.trimf(damage_size.universe, [30, 50, 100])     

    # Membership functions for damage level output
    damage_level_fuzzy.automf(names=['Excellent', 'Good', 'Fair', 'Poor', 'Severe', 'Critical'])

    # Define fuzzy rules based on defect class
    if predicted_class == 'Rỉ sét':
        rule1 = ctrl.Rule(damage_size['Ri0'], damage_level_fuzzy['Excellent'])
        rule2 = ctrl.Rule(damage_size['Ri1'], damage_level_fuzzy['Good'])
        rule3 = ctrl.Rule(damage_size['Ri2'], damage_level_fuzzy['Fair'])
        rule4 = ctrl.Rule(damage_size['Ri3'], damage_level_fuzzy['Poor'])
        rule5 = ctrl.Rule(damage_size['Ri4'], damage_level_fuzzy['Severe'])
        rule6 = ctrl.Rule(damage_size['Ri5'], damage_level_fuzzy['Critical'])

    elif predicted_class == 'Bong tróc sơn':
        # Rules for 'Bong tróc sơn'
        rule1 = ctrl.Rule(damage_size['BT0'], damage_level_fuzzy['Excellent'])
        rule2 = ctrl.Rule(damage_size['BT1'], damage_level_fuzzy['Good'])
        rule3 = ctrl.Rule(damage_size['BT2'], damage_level_fuzzy['Fair'])
        rule4 = ctrl.Rule(damage_size['BT3'], damage_level_fuzzy['Poor'])
        rule5 = ctrl.Rule(damage_size['BT4'], damage_level_fuzzy['Severe'])
        rule6 = ctrl.Rule(damage_size['BT5'], damage_level_fuzzy['Critical'])

    elif predicted_class == 'Vết nứt':
        # Rules for 'Vết nứt'
        rule1 = ctrl.Rule(damage_size['VN0'], damage_level_fuzzy['Excellent'])
        rule2 = ctrl.Rule(damage_size['VN1'], damage_level_fuzzy['Good'])
        rule3 = ctrl.Rule(damage_size['VN2'], damage_level_fuzzy['Fair'])
        rule4 = ctrl.Rule(damage_size['VN3'], damage_level_fuzzy['Poor'])
        rule5 = ctrl.Rule(damage_size['VN4'], damage_level_fuzzy['Severe'])
        rule6 = ctrl.Rule(damage_size['VN5'], damage_level_fuzzy['Critical'])
    
    else:
        # Trường hợp mặc định khi không có điều kiện nào khớp
        rule1 = ctrl.Rule(damage_size['BT0'], damage_level_fuzzy['Excellent'])
        rule2 = ctrl.Rule(damage_size['BT1'], damage_level_fuzzy['Good'])
        rule3 = ctrl.Rule(damage_size['BT2'], damage_level_fuzzy['Fair'])
        rule4 = ctrl.Rule(damage_size['BT3'], damage_level_fuzzy['Poor'])
        rule5 = ctrl.Rule(damage_size['BT4'], damage_level_fuzzy['Severe'])
        rule6 = ctrl.Rule(damage_size['BT5'], damage_level_fuzzy['Critical'])

    # Combine the rules into a control system
    damage_ctrl = ctrl.ControlSystem([rule1, rule2, rule3, rule4, rule5, rule6])
    damage_simulation = ctrl.ControlSystemSimulation(damage_ctrl)

    # Pass the input values to the fuzzy control system
    damage_simulation.input['damage_size'] = damage_percentage

    # Compute the fuzzy logic result
    damage_simulation.compute()
    return damage_simulation.output['damage_level']


def predict_defect_details(image, detected_objects):
    results = []
    
    if not isinstance(detected_objects, list):
        print(f"Detected objects is not a list: {detected_objects}")
        return results 
    
    for obj in detected_objects:
        if not isinstance(obj, dict):
            print(f"Detected object is not a dictionary: {obj}")
            continue  

        print(f"Processing object: {obj}")
        
        try:
            min_x, min_y, max_x, max_y = (
                int(obj.get('x_min', 0)), 
                int(obj.get('y_min', 0)), 
                int(obj.get('x_max', 0)), 
                int(obj.get('y_max', 0))
            )
            
            cropped_image = image[min_y:max_y, min_x:max_x]

            if cropped_image.size == 0:
                continue

            cropped_image_resized = cv2.resize(cropped_image, (128, 128))
            preprocessed_image = preprocess_input(np.expand_dims(cropped_image_resized, axis=0))

            prediction = model.predict(preprocessed_image)
            predicted_label = np.argmax(prediction)
            predicted_class = class_names[predicted_label]
            prediction_confidence = float(np.max(prediction))

            obj['defect_type'] = predicted_class  

            img_width, img_height = image.shape[1], image.shape[0]
            
            # Use the segmentation mask to calculate the damage percentage
            mask = obj.get('mask', np.zeros((img_height, img_width)))  # Default to zero mask if not found
            damage_percentage = float(calculate_damage_level(mask, img_width, img_height))
            avg_color = np.mean(cropped_image, axis=(0, 1))
            color_code = float(np.mean(avg_color))

            print(f"Damage Percentage: {damage_percentage}, Color Code: {color_code}")

            damage_level_result = fuzzy_logic_damage_level(damage_percentage, predicted_class)
            print(f"Fuzzy Logic - Damage Level Result: {damage_level_result}")

            model_cp = cp_model.CpModel()
            damage_var = model_cp.NewIntVar(0, 5, 'damage_level')
            if damage_level_result < 5:
                model_cp.Add(damage_var == 0)  # Không hư hại
            elif 5 <= damage_level_result < 15:
                model_cp.Add(damage_var == 1)  # Hư hại rất nhẹ
            elif 15 <= damage_level_result < 30:
                model_cp.Add(damage_var == 2)  # Hư hại nhẹ
            elif 30 <= damage_level_result < 50:
                model_cp.Add(damage_var == 3)  # Hư hại trung bình
            elif 50 <= damage_level_result < 70:
                model_cp.Add(damage_var == 4)  # Hư hại nặng
            else:
                model_cp.Add(damage_var == 5)  # Hư hại rất nặng

            solver = cp_model.CpSolver()
            status = solver.Solve(model_cp)
            if status == cp_model.OPTIMAL:
                damage_severity = solver.Value(damage_var)
                
                # Định nghĩa hành động cụ thể dựa trên mức độ hư hại
                if damage_severity == 0:
                    action = "Không cần hành động"  # Không hư hại
                elif damage_severity == 1:
                    action = "Giám sát và theo dõi"  # Hư hại rất nhẹ
                elif damage_severity == 2:
                    action = "Thực hiện bảo dưỡng nhỏ"  # Hư hại nhẹ
                elif damage_severity == 3:
                    action = "Bảo dưỡng khẩn cấp"  # Hư hại trung bình
                elif damage_severity == 4:
                    action = "Sửa chữa hoặc thay thế"  # Hư hại nặng
                else:
                    action = "Thay thế toàn bộ"  # Hư hại rất nặng

                    
                results.append({
                    'defect_type': predicted_class,
                    'confidence': prediction_confidence,
                    'damage_level': action,
                    'color_code': color_code,
                    'damage_percentage': damage_percentage,
                    'img_width': img_width,  
                    'img_height': img_height,  
                    'x_min': min_x,
                    'y_min': min_y,
                    'x_max': max_x,
                    'y_max': max_y
                })

        except Exception as e:
            print(f"Error processing object: {e}, Object: {obj}")
            continue
    print("Kết quả cuối cùng sau dự đoán ở result:")
    for obj in results:
        print(obj)

    return results

def get_video_info(video_path):
    # Lấy tên file từ đường dẫn
    video_name = os.path.splitext(os.path.basename(video_path))[0]

    # Open the video file
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        return {"error": "Cannot open video file."}

    # Get frame count, FPS, and duration
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    fps = math.ceil(fps)  # Làm tròn FPS lên
    duration = frame_count / fps if fps > 0 else 0

    cap.release()
    return {
        "name": video_name,
        "frame_count": frame_count,
        "fps": fps,
        "duration": duration
    }

def process_video(video_path, output_folder, model_path, frame_skip=1):
    # Mở video
    video = cv2.VideoCapture(video_path)
    frame_count = 0
    result = []  # Danh sách để lưu kết quả phát hiện

    # Kiểm tra nếu mở video thành công
    if not video.isOpened():
        print("Không thể mở video:", video_path)
        return result

    # Đọc frame đầu tiên
    success, frame = video.read()
    while success:
        # Chỉ xử lý nếu frame_count chia hết cho frame_skip
        if frame_count % frame_skip == 0:
            # Gọi hàm phát hiện và phân đoạn các đối tượng trên frame
            detected_objects = detect_and_segment_objects(frame, model_path)
            
            # Danh sách để lưu kết quả của các đối tượng trên frame hiện tại
            frame_data = {"frame_id": int(frame_count), "objects": []}

            # Vẽ các đối tượng đã phát hiện lên frame và lưu thông tin vào frame_data["objects"]
            for obj in detected_objects:
                required_keys = ('x_min', 'y_min', 'x_max', 'y_max', 'class_name', 'confidence', 'mask', 'img_width', 'img_height')
                
                if all(k in obj for k in required_keys):
                    class_name = obj['class_name']
                    confidence = float(obj['confidence'])
                    mask = obj['mask']  # mask là ảnh nhị phân chứa segmentation của đối tượng

                    # Vẽ mask segmentation lên frame
                    if mask is not None:
                        # Tạo một vùng màu cho mỗi class
                        color = np.random.randint(0, 255, (1, 3), dtype=np.uint8)[0].tolist()
                        
                        # Resize mask cho đúng kích thước với frame nếu cần
                        resized_mask = cv2.resize(mask, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_NEAREST)
                        
                        # Tạo một lớp màu cho mask
                        mask_colored = np.zeros_like(frame, dtype=np.uint8)
                        mask_colored[resized_mask > 0] = color
                        
                        # Kết hợp mask vào frame
                        frame = cv2.addWeighted(frame, 1, mask_colored, 0.5, 0)
                    
                    # Thêm thông tin đối tượng vào frame_data["objects"]
                    frame_data["objects"].append({
                        "class_name": class_name,
                        "confidence": confidence,
                    })
            
            # Thêm thông tin frame vào danh sách kết quả
            result.append(frame_data)

            # Lưu frame với đối tượng đã phát hiện vào output_folder
            output_frame_path = os.path.join(output_folder, f"frame_{frame_count}.png")
            cv2.imwrite(output_frame_path, frame)
        
        # Tăng biến đếm frame và đọc frame tiếp theo
        frame_count += 1
        success, frame = video.read()
    
    # Giải phóng video sau khi hoàn tất xử lý
    video.release()
    print(f"Video được xử lý xong, các frame lưu tại: {output_folder}")

    # Lưu kết quả phát hiện dưới dạng JSON
    result_file_path = os.path.join(output_folder, "detection_results.json")
    with open(result_file_path, "w") as result_file:
        json.dump(result, result_file, indent=4)

    print(f"Kết quả phát hiện lưu tại: {result_file_path}")

    return result_file_path


def process_and_predict_from_video_output(video_output_folder, model_path, server_url="http://127.0.0.1:5000"):
    # List to hold the results
    all_results = []

    # Get the list of image files in the video output folder
    frame_files = [f for f in os.listdir(video_output_folder) if f.endswith(".png")]
    
    # Sort the frame files to ensure processing in the correct order
    frame_files.sort(key=lambda x: int(x.split('_')[1].split('.')[0]))  # Assuming frame file names like 'frame_1.png'

    # Process each frame
    for frame_file in frame_files:
        frame_path = os.path.join(video_output_folder, frame_file)
        frame = cv2.imread(frame_path)

        if frame is not None:
            # Call the predict_defect_details function to predict defects on the current frame
            detected_objects = detect_and_segment_objects(frame, model_path)  # This would already include the segmentations
            frame_results = predict_defect_details(frame, detected_objects)

            # For each detected object in the frame, extract the necessary details
            for detected_object in frame_results:
                # Extract the details you want for each object
                defect_type = detected_object.get('defect_type', '')
                confidence = detected_object.get('confidence', 0)
                damage_level = detected_object.get('damage_level', '')
                color_code = detected_object.get('color_code', '')
                damage_percentage = detected_object.get('damage_percentage', 0)
                x_min = detected_object.get('x_min', 0)
                y_min = detected_object.get('y_min', 0)
                x_max = detected_object.get('x_max', 0)
                y_max = detected_object.get('y_max', 0)

                # Get image dimensions
                img_height, img_width, _ = frame.shape

                # Construct the frame URL
                frame_url = f"{server_url}/predicted_images/{frame_file}"

                # Collect all the results for the frame
                all_results.append({
                    "frame_id": frame_file.split('.')[0],  # Frame ID (from filename)
                    "frame_url": frame_url,  # URL của frame
                    "defect_type": defect_type,
                    "confidence": confidence,
                    "damage_level": damage_level,
                    "color_code": color_code,
                    "damage_percentage": damage_percentage,
                    "img_width": img_width,
                    "img_height": img_height,
                    "x_min": x_min,
                    "y_min": y_min,
                    "x_max": x_max,
                    "y_max": y_max
                })

    # Define a priority order for damage levels
    DAMAGE_LEVEL_PRIORITY = {
        "Không cần hành động":1,
        "Giám sát và theo dõi": 2,
        "Thực hiện bảo dưỡng nhỏ": 3, 
        "Bảo dưỡng khẩn cấp": 4,
        "Sửa chữa hoặc thay thế": 5,  
        "Thay thế toàn bộ": 6 
    }


    # Step 1: Aggregate results by frame_id and defect_type
    # Step 1: Aggregate results by frame_id and defect_type
    summary = {}
    for result in all_results:
        frame_id = result["frame_id"]
        frame_url = result["frame_url"]  # Get the URL for each frame
        defect_type = result["defect_type"]
        damage_percentage = result["damage_percentage"]
        damage_level = result["damage_level"]

        # Initialize frame_id in summary if not present
        if frame_id not in summary:
            summary[frame_id] = {
                "frame_url": frame_url,  # Add frame URL to the summary
                "total_damage_by_type": {},
                "max_damage_type": None,
                "max_damage_percentage": 0,
                "damage_levels": {},  # Store damage levels by defect type
                "most_severe_damage_level": None  # Store the most severe damage level for the frame
            }

        # Initialize defect_type in total_damage_by_type for the frame_id if not present
        if defect_type not in summary[frame_id]["total_damage_by_type"]:
            summary[frame_id]["total_damage_by_type"][defect_type] = 0
            summary[frame_id]["damage_levels"][defect_type] = []  # Initialize list of damage levels for the defect type

        # Accumulate damage_percentage for the defect_type in this frame_id
        summary[frame_id]["total_damage_by_type"][defect_type] += damage_percentage

        # Append the damage_level to the defect_type's list
        if damage_level:
            summary[frame_id]["damage_levels"][defect_type].append(damage_level)

    # Step 2: Find the defect_type with the highest damage_percentage and the most severe damage_level
    for frame_id, details in summary.items():
        max_damage_type = None
        max_damage_percentage = 0
        most_severe_damage_level = None

        for defect_type, total_damage in details["total_damage_by_type"].items():
            # Find the defect_type with the highest damage percentage
            if total_damage > max_damage_percentage:
                max_damage_type = defect_type
                max_damage_percentage = total_damage

            # Find the most severe damage_level across all defect types
            for level in details["damage_levels"][defect_type]:
                if most_severe_damage_level is None or DAMAGE_LEVEL_PRIORITY[level] > DAMAGE_LEVEL_PRIORITY[most_severe_damage_level]:
                    most_severe_damage_level = level

        # Store the maximum damage defect_type and percentage in the summary
        summary[frame_id]["max_damage_type"] = max_damage_type
        summary[frame_id]["max_damage_percentage"] = max_damage_percentage
        summary[frame_id]["most_severe_damage_level"] = most_severe_damage_level


    # Return the collected results and the summary
    return all_results, summary
