import React, { useState } from "react";
import {
  Button,
  Spin,
  Typography,
  message,
  Upload,
  Slider,
  Progress,
} from "antd";
import modelService from "../utils/client";
import { useAutoAnimate } from "@formkit/auto-animate/react";
import Lottie from "lottie-react";
import Animation from "../assets/Animation.json";
import { UploadOutlined } from "@ant-design/icons";
import "./index.css";

const { Title } = Typography;

export default function PredictImage() {
  const [autoAnimation] = useAutoAnimate();
  const [loadingPrediction, setLoadingPrediction] = useState(false);
  const [predictions, setPredictions] = useState([]);
  const [allPredictions, setAllPredictions] = useState([]); // Thêm state để lưu kết quả gốc
  const [damageSummary, setDamageSummary] = useState({});
  const [messageApi, contextHolder] = message.useMessage({ duration: 5 });
  const [selectedImage, setSelectedImage] = useState(null);
  const [selectedFileName, setSelectedFileName] = useState("");
  const [evaluationMessage, setEvaluationMessage] = useState("");
  const [confidenceThreshold, setConfidenceThreshold] = useState(0.6); // State cho Slider
  const [filteredCount, setFilteredCount] = useState(0); // State cho tổng số ảnh dự đoán sau lọc

  const handleImageChange = (info) => {
    const latestFile = info.fileList[info.fileList.length - 1];

    if (latestFile && latestFile.originFileObj) {
      setSelectedImage(latestFile.originFileObj);
      setSelectedFileName(latestFile.name);
    }

    if (info.file.status === "done") {
      message.success(`${info.file.name} đã được tải lên thành công.`);
    } else if (info.file.status === "error") {
      message.error(`${info.file.name} tải lên thất bại.`);
    }
  };

  const handlePredict = async () => {
    if (!selectedImage) {
      message.error("Vui lòng chọn một hình ảnh để dự đoán.");
      return;
    }

    setLoadingPrediction(true);
    messageApi.open({
      type: "loading",
      content: "Đang dự đoán...",
      duration: 0,
    });

    const formData = new FormData();
    formData.append("image", selectedImage);

    try {
      const response = await modelService.predict(formData);
      const { results } = response;

      if (results.length > 0) {
        const newDamageSummary = {};

        // Lưu toàn bộ kết quả mà không lọc
        setAllPredictions(results);

        // Lọc kết quả lần đầu theo ngưỡng confidenceThreshold
        const initialFilteredPredictions = results
          .filter((result) => result.confidence >= confidenceThreshold)
          .map((result, index) => {
            const {
              defect_type,
              confidence,
              damage_level,
              color_code,
              damage_percentage,
              img_width,
              img_height,
              x_min,
              y_min,
              x_max,
              y_max,
            } = result;

            const bbox_width = x_max - x_min;
            const bbox_height = y_max - y_min;
            const defect_area = bbox_width * bbox_height;
            const original_area = img_width * img_height;
            const damage_area_percentage = (
              (defect_area / original_area) *
              100
            ).toFixed(2);

            if (!newDamageSummary[defect_type]) {
              newDamageSummary[defect_type] = 0;
            }
            newDamageSummary[defect_type] += parseFloat(damage_area_percentage);

            const imageUrl = `http://127.0.0.1:5000/predicted_images/${selectedImage.name.replace(
              /\.[^/.]+$/,
              ""
            )}_predicted_${index}.png`;

            return {
              imageUrl,
              label: (
                <div>
                  <p className="defect-type">Loại lỗi: {defect_type}</p>
                  <p className="confidence">
                    Độ tin cậy: {(confidence * 100).toFixed(2)}%
                  </p>
                  <p className="damage-size">
                    Diện tích hư hại: {damage_area_percentage}%
                  </p>
                  <p className="color-code">
                    Đề nghị: <br />
                    {damage_level}
                  </p>
                </div>
              ),
              fileName: `${selectedImage.name}_predicted_${index}.jpg`,
            };
          });

        setPredictions(initialFilteredPredictions);
        setDamageSummary(newDamageSummary);
        setFilteredCount(initialFilteredPredictions.length); // Cập nhật tổng số ảnh dự đoán sau khi lọc lần đầu
      }
    } catch (error) {
      message.error("Đã xảy ra lỗi trong quá trình dự đoán.");
    } finally {
      setLoadingPrediction(false);
      messageApi.destroy();
    }
  };

  const filterPredictions = (threshold) => {
    const filtered = allPredictions
      .filter((result) => result.confidence >= threshold)
      .map((result, index) => {
        const {
          defect_type,
          confidence,
          damage_level,
          color_code,
          damage_percentage,
          img_width,
          img_height,
          x_min,
          y_min,
          x_max,
          y_max,
        } = result;

        const bbox_width = x_max - x_min;
        const bbox_height = y_max - y_min;
        const defect_area = bbox_width * bbox_height;
        const original_area = img_width * img_height;
        const damage_area_percentage = (
          (defect_area / original_area) *
          100
        ).toFixed(2);

        const imageUrl = `http://127.0.0.1:5000/predicted_images/${selectedImage.name.replace(
          /\.[^/.]+$/,
          ""
        )}_predicted_${index}.png`;

        return {
          imageUrl,
          label: (
            <div>
              <p className="defect-type">Loại lỗi: {defect_type}</p>
              <p className="confidence">
                Độ tin cậy: {(confidence * 100).toFixed(2)}%
              </p>
              <p className="damage-size">
                Diện tích hư hại: {damage_area_percentage}%
              </p>
              <p className="color-code">
                Đề nghị: <br />
                {damage_level}
              </p>
            </div>
          ),
          fileName: `${selectedImage.name}_predicted_${index}.jpg`,
        };
      });
    setPredictions(filtered);
    setFilteredCount(filtered.length);
  };

  // Gọi filterPredictions khi giá trị slider thay đổi
  const handleSliderChange = (value) => {
    const threshold = value / 100; // Chuyển đổi về giá trị từ 0 đến 1
    setConfidenceThreshold(threshold);
    filterPredictions(threshold); // Lọc lại kết quả khi slider thay đổi
  };

  const handleEvaluate = () => {
    messageApi.open({
      type: "loading",
      content: "Đang đánh giá...",
      duration: 3,
    });

    setTimeout(() => {
      messageApi.destroy();
      message.success("Đánh giá hoàn tất.");

      // Tính tổng damage_percentage cho từng loại lỗi (defect_type) dựa trên dữ liệu trong predictions
      const damageTotals = predictions.reduce((acc, prediction) => {
        const { label } = prediction;

        // Lấy `defect_type` và `damage_percentage` từ `label`
        const defectTypeMatch = label.props.children.find(
          (child) =>
            child && child.props && child.props.className === "defect-type"
        );
        const damageSizeMatch = label.props.children.find(
          (child) =>
            child && child.props && child.props.className === "damage-size"
        );

        if (defectTypeMatch && damageSizeMatch) {
          const defectType = defectTypeMatch.props.children[1];
          const damageAreaPercentage = parseFloat(
            damageSizeMatch.props.children[1]
          );

          // Tính tổng diện tích hư hại cho từng loại lỗi
          if (!acc[defectType]) {
            acc[defectType] = 0;
          }
          acc[defectType] += damageAreaPercentage;
        }

        return acc;
      }, {});

      setDamageSummary(damageTotals);

      // Hiển thị kết quả đánh giá
      setEvaluationMessage(
        <div className="evaluation-container">
          <p className="evaluation-title">Kết quả đánh giá:</p>
          <div className="evaluation-grid">
            {Object.entries(damageTotals).map(([defectType, totalDamage]) => {
              let recommendation = "Không cần thực hiện biện pháp sửa chữa."; // Khởi tạo giá trị mặc định

              if (defectType === "Rỉ sét" && totalDamage > 20) {
                recommendation = "Khuyến nghị áp dụng giải pháp thay thế.";
              } else if (defectType === "Bong tróc sơn" && totalDamage > 60) {
                recommendation = "Khuyến nghị sơn lại.";
              }

              // Chọn màu cho Progress dựa trên recommendation
              const progressColor =
                recommendation === "Không cần thực hiện biện pháp sửa chữa."
                  ? "#52c41a" // Xanh lá
                  : "#faad14"; // Vàng

              return (
                <>
                  <div className="evaluation-defect">{defectType}</div>
                  <div className="evaluation-recommendation">
                    {recommendation}
                  </div>
                  <div className="evaluation-progress">
                    <Progress
                      type="circle"
                      percent={totalDamage.toFixed(1)}
                      format={(percent) => `${percent}%`}
                      strokeColor={progressColor} // Áp dụng màu cho Progress
                      width={60}
                    />
                  </div>
                </>
              );
            })}
          </div>
        </div>
      );
    }, 3000);
  };

  return (
    <div>
      {contextHolder}
      <div className="grid grid-cols-12">
        <div className="col-span-5 mx-auto">
          <Lottie animationData={Animation} className="w-[300px]" />
        </div>
        <div className="col-span-7">
          <p className="text-xl mt-10 my-5 bg-clip-text text-transparent bg-gradient-to-l from-blue-900 to-blue-500">
            Hệ thống nhận diện vết hao mòn bằng AI{" "}
          </p>
          <p className="text-xl mt-5 bg-clip-text text-transparent bg-gradient-to-l from-blue-900 to-blue-500">
            Đánh giá và đề xuất biện pháp sửa chữa <br />
            dựa theo tiêu chuẩn <strong>ISO 4628*</strong>{" "}
          </p>
          <p className="text-l italic mt-5 bg-clip-text text-transparent bg-gradient-to-l from-blue-900 to-blue-500">
            <strong>ISO 4628:</strong> Tiêu chuẩn quốc gia TCVN 12005-1:2017 về
            Sơn và vec ni – <br /> Đánh giá sự suy biến của lớp phủ{" "}
          </p>
        </div>
      </div>
      <div className="grid grid-cols-12">
        <div className="col-span-5 mr-2">
          <p className="text-xl italic bg-clip-text text-transparent bg-gradient-to-l from-blue-900 to-blue-500">
            Chọn ảnh:
          </p>
          <Upload
            name="image"
            listType="picture"
            maxCount={1}
            onChange={handleImageChange}
            beforeUpload={() => false}
          >
            <Button icon={<UploadOutlined />}>Chọn ảnh</Button>
          </Upload>

          <Button
            onClick={handlePredict}
            disabled={loadingPrediction}
            className="mt-5"
          >
            Nhận diện
          </Button>
          <div className="col-span-2 mt-4">
            <p className="text-l italic bg-clip-text text-transparent bg-gradient-to-l from-blue-900 to-blue-500">
              Tùy chỉnh độ chính xác:
            </p>
            <Slider
              defaultValue={60}
              min={0}
              max={100}
              onChange={handleSliderChange}
              tooltip={{ formatter: (value) => `${value}%` }}
            />
          </div>
        </div>

        <div className="col-span-7" ref={autoAnimation}>
          <div className="my-5">
            {loadingPrediction && (
              <div className="flex justify-center">
                <Spin />
              </div>
            )}
            {predictions.length > 0 && !loadingPrediction && (
              <div className="flex flex-col background-predict">
                <Title level={3}>Kết quả dự đoán:</Title>
                <p>
                  Vết hao mòn phát hiện được: <strong>{filteredCount}</strong>
                </p>
                <div className="grid grid-cols-3 gap-4 scrollable-container">
                  {predictions.map((prediction, index) => (
                    <div key={index} className="flex flex-col items-center">
                      {prediction.imageUrl ? (
                        <div className="image-container">
                          <img
                            src={prediction.imageUrl}
                            alt={`Kết quả ${index + 1}`}
                            className="predicted-image"
                            onError={() => {
                              console.error(
                                `Không thể tải hình ảnh từ URL: ${prediction.imageUrl}`
                              );
                              message.error(
                                `Không thể tải hình ảnh: ${prediction.fileName}`
                              );
                            }}
                          />
                        </div>
                      ) : (
                        <p>Không có hình ảnh dự đoán.</p>
                      )}
                      <Title level={5}>
                        Dự đoán: <strong>{prediction.label}</strong>
                      </Title>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>
        </div>
      </div>
      {predictions.length > 0 && (
        <Button onClick={handleEvaluate} className="mt-5">
          Đánh giá
        </Button>
      )}

      {evaluationMessage && evaluationMessage}
    </div>
  );
}
