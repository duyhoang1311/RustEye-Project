import React, { useState } from "react";
import {
  Button,
  Upload,
  Typography,
  message,
  Spin,
  InputNumber,
  Table,
  Progress,
} from "antd";
import { PlusOutlined } from "@ant-design/icons";
import modelService from "../utils/client";
import "./index.css";

const { Title, Text } = Typography;

export default function PredictVideo() {
  const [selectedFile, setSelectedFile] = useState(null);
  const [videoURL, setVideoURL] = useState(null);
  const [videoInfo, setVideoInfo] = useState(null);
  const [loading, setLoading] = useState(false);
  const [predictionResults, setPredictionResults] = useState(null);
  const [frameStep, setFrameStep] = useState(500);
  const [suggestedSteps, setSuggestedSteps] = useState([]);
  const [showEvaluation, setShowEvaluation] = useState(false);

  // Handle video file selection
  const handleFileChange = (info) => {
    const latestFile = info.fileList[info.fileList.length - 1];
    if (latestFile && latestFile.originFileObj.type.startsWith("video")) {
      setSelectedFile(latestFile.originFileObj);
      const url = URL.createObjectURL(latestFile.originFileObj);
      setVideoURL(url);
      message.success(`${latestFile.name} đã được chọn.`);
    } else {
      message.error("Vui lòng chọn một file video hợp lệ.");
    }
  };

  // Handle getting video information
  const handleGetVideoInfo = async () => {
    if (!selectedFile) {
      message.warning("Chưa chọn file video.");
      return;
    }

    setLoading(true);
    try {
      const response = await modelService.getVideoInfo(selectedFile);
      setVideoInfo(response);

      // Tạo gợi ý frame step dựa trên tổng số frame và FPS
      if (response.frame_count && response.fps) {
        const maxFrames = response.frame_count;
        const fps = response.fps;

        // Tính toán các giá trị gợi ý
        const suggestions = [
          Math.floor(fps), // Mỗi giây
          Math.floor(fps * 2), // Mỗi 2 giây
          Math.floor(maxFrames / 10),
          Math.floor(maxFrames / 5),
          Math.floor(maxFrames / 3),
        ]
          .filter((step) => step < maxFrames)
          .slice(0, 5); // Chỉ giữ 2–5 giá trị gợi ý đầu tiên

        const roundedSteps = suggestions.map(
          (step) => Math.floor(step / 10) * 10
        );

        const uniqueSteps = [...new Set(roundedSteps)].filter(
          (step) => step < maxFrames
        );

        setSuggestedSteps(uniqueSteps.slice(0, 5));
      }

      message.success("Lấy thông tin video thành công!");
    } catch (error) {
      console.error("Lỗi khi lấy thông tin video:", error);
      message.error("Không thể lấy thông tin video.");
    } finally {
      setLoading(false);
    }
  };

  const handlePredict = async () => {
    if (!selectedFile) {
      message.warning("Chưa chọn file video.");
      return;
    }

    setLoading(true);
    setPredictionResults(null);

    try {
      const response = await modelService.predictVideo(selectedFile, frameStep);
      setPredictionResults(response.summary);
      message.success("Nhận diện hoàn tất!");
    } catch (error) {
      console.error("Lỗi khi nhận diện video:", error);
      message.error("Không thể nhận diện video.");
    } finally {
      setLoading(false);
    }
  };

  const handleFrameStepChange = (value) => {
    setFrameStep(value);
  };

  // Hàm để hiển thị bảng đánh giá
  const handleShowEvaluation = () => {
    setShowEvaluation(true);
  };

  // Tạo dữ liệu tổng hợp
  const evaluationData = Object.entries(predictionResults || {}).map(
    ([frameId, frameData]) => ({
      key: frameId,
      damageLevel: frameData.most_severe_damage_level || "Không có thông tin",
    })
  );

  // Tính số lần xuất hiện của từng đề nghị và tỷ lệ phần trăm
  const aggregatedData = evaluationData.reduce((acc, { damageLevel }) => {
    if (!acc[damageLevel]) {
      acc[damageLevel] = { damageLevel, frameCount: 0 };
    }
    acc[damageLevel].frameCount += 1;
    return acc;
  }, {});

  const totalFrames = Object.keys(predictionResults || {}).length;

  const aggregatedArray = Object.values(aggregatedData).map(
    ({ damageLevel, frameCount }) => ({
      damageLevel,
      percentage: ((frameCount / totalFrames) * 100).toFixed(2),
      frameCount,
    })
  );

  // Cấu hình cột
  const columns = [
    {
      title: "Đề nghị",
      dataIndex: "damageLevel",
      key: "damageLevel",
    },
    {
      title: "Phần trăm",
      dataIndex: "percentage",
      key: "percentage",
      render: (percentage) => (
        <Progress
          percent={Number(percentage)}
          status={
            percentage < 50
              ? "normal"
              : percentage < 80
              ? "active"
              : "exception"
          }
          strokeColor={
            percentage < 50
              ? "#52c41a"
              : percentage < 80
              ? "#faad14"
              : "#f5222d"
          }
        />
      ),
    },
  ];

  return (
    <div style={{ padding: 20 }}>
      <Title level={3}>Tải lên và dự đoán Video</Title>
      <Text>
        Vui lòng chọn tệp video và nhấp vào "Nhận diện" sau đó nhấp vào "Xem
        thông tin" để phân tích video.
      </Text>

      <div className="mt-2">
        <Upload
          beforeUpload={() => false}
          onChange={handleFileChange}
          maxCount={1}
          listType="picture-card"
        >
          {selectedFile ? null : (
            <div>
              <PlusOutlined />
              <div style={{ marginTop: 8 }}>Upload</div>
            </div>
          )}
        </Upload>
      </div>
      <div className="grid grid-cols-12">
        <div className="col-span-5">
          {videoURL && (
            <div className="mt-2">
              <video width="500" height="300" controls>
                <source src={videoURL} type="video/mp4" />
                Your browser does not support the video tag.
              </video>
            </div>
          )}
          {videoInfo && (
            <div className="flex flex-col">
              <Title level={4}>Thông tin video:</Title>
              <p>Tổng số frame: {videoInfo.frame_count}</p>
              <p>FPS: {videoInfo.fps}</p>
              <p>Thời lượng: {videoInfo.duration.toFixed(2)} giây</p>
            </div>
          )}

          <div className="mt-2">
            <Text>Chọn số frame step:</Text>
            <InputNumber
              min={1}
              max={videoInfo?.frame_count || undefined}
              defaultValue={500}
              value={frameStep}
              onChange={handleFrameStepChange}
              style={{ marginLeft: 10 }}
            />

            {suggestedSteps.length > 0 && (
              <div style={{ marginTop: 10 }}>
                <Text>Gợi ý: </Text>
                {suggestedSteps.map((step, index) => (
                  <Button
                    key={index}
                    style={{ margin: "0 5px" }}
                    onClick={() => setFrameStep(step)} // Cập nhật giá trị frame step khi nhấn nút
                  >
                    {step}
                  </Button>
                ))}
              </div>
            )}
          </div>

          <div className="mt-2 flex space-x-4">
            <Button
              type="primary"
              onClick={handleGetVideoInfo}
              disabled={!selectedFile}
            >
              Xem thông tin
            </Button>
            <Button
              type="primary"
              onClick={handlePredict}
              disabled={!selectedFile}
            >
              Nhận diện
            </Button>
          </div>
        </div>
        <div className="col-span-7">
          <div className="my-3">
            <div className="flex justify-center">
              {loading && (
                <Spin style={{ marginTop: 20 }} tip="Đang xử lý..." />
              )}
            </div>

            {predictionResults && (
              <div className="flex flex-col background-predict">
                <Title level={3}>Kết quả dự đoán:</Title>
                <p>
                  Số frame đã kiểm tra:{" "}
                  <strong>{Object.keys(predictionResults).length}</strong>
                </p>
                <div className="grid grid-cols-3 gap-4 scrollable-container">
                  {Object.entries(predictionResults).map(
                    ([frameId, frameData], index) => (
                      <div key={frameId} className="flex flex-col items-center">
                        <div className="image-container">
                          <img
                            src={`http://127.0.0.1:5000/predicted_images/${videoInfo?.name}.MOV_frames/${frameId}.png`}
                            alt={`Frame ${frameId}`}
                            className="predicted-image"
                          />
                        </div>
                        <Title level={5}>Frame ID: {frameId}</Title>
                        <div>
                          <p className="damage-level">
                            <strong>Đề nghị:</strong>{" "}
                            {frameData.most_severe_damage_level ||
                              "Không có thông tin"}
                          </p>
                          <p className="confidence">Phần trăm hao mòn:</p>
                          <ul className="confidence-list">
                            {Object.entries(frameData.total_damage_by_type).map(
                              ([damageType, damageValue]) => (
                                <li key={damageType}>
                                  {damageType}: {damageValue.toFixed(2)}%
                                </li>
                              )
                            )}
                          </ul>
                        </div>
                      </div>
                    )
                  )}
                </div>
                {/* Nút "Đánh giá" */}
                <Button
                  type="primary"
                  onClick={handleShowEvaluation}
                  style={{ marginTop: 20 }}
                >
                  Đánh giá
                </Button>
              </div>
            )}
          </div>
        </div>
      </div>

      {showEvaluation && (
        <div style={{ marginTop: 20 }}>
          <Title level={4}>Bảng đánh giá:</Title>
          <Table
            dataSource={aggregatedArray}
            columns={columns}
            pagination={false}
            size="middle"
          />
        </div>
      )}
    </div>
  );
}
