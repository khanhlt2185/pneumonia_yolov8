/**
 * frontend/src/services/api.js
 * Quản lý kết nối API tới Backend FastAPI thông qua axios.
 */

import axios from 'axios';

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000/api/v1';

// Tạo axios client với timeout và header mặc định
const apiClient = axios.create({
  baseURL: API_BASE_URL,
  timeout: 30000,
});

/**
 * Kiểm tra trạng thái máy chủ Backend.
 */
export const fetchHealth = async () => {
  const response = await apiClient.get('/health');
  return response.data;
};

/**
 * Lấy danh sách 4-6 ảnh mẫu thử nghiệm từ server.
 */
export const fetchSamples = async () => {
  const response = await apiClient.get('/samples');
  return response.data;
};

/**
 * Tải một file ảnh mẫu từ backend dưới dạng Blob.
 * @param {string} relativeUrl - Ví dụ: /static/samples/sample_1_...
 */
export const fetchImageBlob = async (relativeUrl) => {
  // Lấy gốc server (bỏ /api/v1)
  const serverOrigin = API_BASE_URL.replace(/\/api\/v1\/?$/, '');
  const fullUrl = `${serverOrigin}${relativeUrl}`;
  const response = await axios.get(fullUrl, { responseType: 'blob' });
  return response.data;
};

/**
 * Gửi ảnh lên backend để thực hiện suy luận mô hình.
 * @param {File|Blob} fileOrBlob - File hoặc Blob của ảnh X-quang
 * @param {string} filename - Tên file
 */
export const predictImage = async (fileOrBlob, filename = 'xray.jpeg') => {
  const formData = new FormData();
  formData.append('file', fileOrBlob, filename);

  const response = await apiClient.post('/predict', formData);
  return response.data;
};

export default apiClient;
