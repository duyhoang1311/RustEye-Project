import "./index.css";
import Animation from "../public/lotties/Animation copy";
import Lottie from "lottie-react";
import { Tabs } from "antd";
import { useMemo } from "react";
import PredictVideo from "./components/PredictVideo";
import PredictImage from "./components/PredictImage";
function App() {
  const tabs = useMemo(() => {
    return [
      {
        key: "1",
        label: "Dự đoán trên hình ảnh",
        children: <PredictImage />,
      },
      {
        key: "2",
        label: "Dự đoán trên video",
        children: <PredictVideo />,
      },
    ];
  }, []);

  return (
    <>
      <header className="w-full bg-gradient-to-b from-indigo-700 to-blue-400 shadow-pink-500/30 px-5">
        <div className="grid grid-cols-12 gap-4">
          <div className="col-span-3">
            <div className="w-[250px]">
              <Lottie animationData={Animation} className="w-full" />
            </div>
          </div>
          <div className="col-span-7 flex flex-col">
            <h1 className="text-5xl mt-[20px] mb-[10px] font-bold text-white">
              RustEye
            </h1>
            <h1 className="text-2xl font-bold mb-6 text-white">
              Fuzzy-Constrained System for Surface Degradation Detection and
              Evaluation on Telecom Poles via Image Segmentation
            </h1>
            <i className="text-xl text-white">
              <i>
                Hệ thống phát hiện và đánh giá hao mòn bề mặt trên trụ cột viễn
                thông với ràng buộc mờ dựa trên phân đoạn ảnh
              </i>
            </i>
          </div>
          <div className="col-span-2 flex items-center justify-center">
            <img
              src="/lotties/logo.jpg"
              className="w-full h-auto max-w-[200px] object-contain"
            />
          </div>
        </div>
      </header>
      <div className="container max-w-[1280px] w-[95%] mx-auto my-3">
        <Tabs animated defaultActiveKey="1" centered items={tabs} />
      </div>
    </>
  );
}

export default App;
