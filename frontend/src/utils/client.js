import axios from "axios";

class ModelService {
  client = null;

  constructor() {
    this.client = axios.create({
      baseURL: "http://127.0.0.1:5000", // Đổi URL tùy theo địa chỉ server của bạn
    });

    // Thêm interceptor để xử lý phản hồi từ API
    this.client.interceptors.response.use(
      (response) => {
        return response.data; // Trả về dữ liệu từ phản hồi
      },
      (error) => {
        console.error("Yêu cầu thất bại:", error);
        throw error.response?.data || error; // Ném lỗi để xử lý ở các nơi gọi hàm
      }
    );
  }

  // Hàm upload hình ảnh
  async uploadImage(imageFile) {
    const formData = new FormData();
    formData.append("image", imageFile);

    try {
      const response = await this.client.post("/upload", formData, {
        headers: {
          "Content-Type": "multipart/form-data",
        },
      });
      return response; // Trả về phản hồi từ API
    } catch (error) {
      console.error("Lỗi khi upload hình ảnh:", error);
      throw error; // Ném lỗi để có thể xử lý ở nơi khác
    }
  }

  // Hàm gọi API để dự đoán
  async predict(formData) {
    try {
      const response = await this.client.post("/predict", formData, {
        headers: {
          "Content-Type": "multipart/form-data",
        },
      });
      return response; // Trả về kết quả dự đoán từ API
    } catch (error) {
      console.error("Lỗi dự đoán:", error);
      throw error; // Ném lỗi để có thể xử lý ở nơi khác
    }
  }

  // Hàm lấy hình ảnh đã được dự đoán từ server
  async getPredictedImage(imageUrl) {
    try {
      const response = await axios.get(imageUrl, {
        responseType: "arraybuffer",
      });
      const base64String = btoa(
        new Uint8Array(response.data).reduce(
          (data, byte) => data + String.fromCharCode(byte),
          ""
        )
      );
      return `data:image/jpeg;base64,${base64String}`; // Hoặc định dạng hình ảnh phù hợp
    } catch (error) {
      console.error("Lỗi khi lấy hình ảnh dự đoán:", error);
      throw new Error("Không thể lấy hình ảnh từ URL.");
    }
  }

  async getVideoInfo(videoFile) {
    const formData = new FormData();
    formData.append("file", videoFile);

    try {
      const response = await this.client.post("/upload-video", formData, {
        headers: {
          "Content-Type": "multipart/form-data",
        },
      });
      return response; // Trả về thông tin video từ API
    } catch (error) {
      console.error("Lỗi khi lấy thông tin video:", error);
      throw error;
    }
  }

  async predictVideo(videoFile, frameStep) {
    const formData = new FormData();
    formData.append("video", videoFile);
    formData.append("frame_step", frameStep);

    try {
      // Gửi yêu cầu POST với axios
      const response = await this.client.post("/predict_video", formData, {
        headers: {
          "Content-Type": "multipart/form-data",
        },
      });
      return response; // Trả về dữ liệu phản hồi từ server
    } catch (error) {
      console.error("Lỗi khi nhận diện video:", error);
      throw error; // Ném lỗi để có thể xử lý ở nơi gọi hàm
    }
  }
}

export default new ModelService();
