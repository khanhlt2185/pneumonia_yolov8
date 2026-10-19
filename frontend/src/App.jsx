import React, { useState, useEffect, useRef } from 'react';
import Header from './components/Header';
import DisclaimerBanner from './components/DisclaimerBanner';
import ImageUpload from './components/ImageUpload';
import SampleGallery from './components/SampleGallery';
import PredictionResult from './components/PredictionResult';
import Toast from './components/Toast';
import { fetchHealth, fetchSamples, fetchImageBlob, predictImage } from './services/api';

export default function App() {
  const [health, setHealth] = useState(null);
  const [isCheckingHealth, setIsCheckingHealth] = useState(true);
  const [samples, setSamples] = useState([]);
  const [isLoadingSamples, setIsLoadingSamples] = useState(true);

  const [selectedFile, setSelectedFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState(null);
  const [selectedSampleId, setSelectedSampleId] = useState(null);

  const [isLoadingPrediction, setIsLoadingPrediction] = useState(false);
  const [predictionResult, setPredictionResult] = useState(null);
  const [toast, setToast] = useState(null);

  const previewUrlRef = useRef(null);

  // Đảm bảo thu hồi URL.revokeObjectURL
  const updatePreviewUrl = (newUrl, isBlobObject = true) => {
    if (previewUrlRef.current && previewUrlRef.current.startsWith('blob:')) {
      URL.revokeObjectURL(previewUrlRef.current);
    }
    previewUrlRef.current = newUrl;
    setPreviewUrl(newUrl);
  };

  useEffect(() => {
    return () => {
      if (previewUrlRef.current && previewUrlRef.current.startsWith('blob:')) {
        URL.revokeObjectURL(previewUrlRef.current);
      }
    };
  }, []);

  // Khởi tạo kiểm tra health và nạp danh sách ảnh mẫu
  useEffect(() => {
    const initApp = async () => {
      // 1. Kiểm tra Health
      try {
        const healthData = await fetchHealth();
        setHealth(healthData);
      } catch (err) {
        console.error('Health check failed:', err);
        setToast({
          message: 'Không thể kết nối tới máy chủ Backend (cổng 8000). Vui lòng đảm bảo server đã chạy.',
          type: 'error',
        });
      } finally {
        setIsCheckingHealth(false);
      }

      // 2. Nạp ảnh mẫu
      try {
        const samplesData = await fetchSamples();
        if (samplesData?.success && samplesData.samples) {
          setSamples(samplesData.samples);
        }
      } catch (err) {
        console.error('Fetch samples failed:', err);
      } finally {
        setIsLoadingSamples(false);
      }
    };

    initApp();
  }, []);

  // Hàm thực thi suy luận
  const executePrediction = async (fileOrBlob, filename) => {
    setIsLoadingPrediction(true);
    setPredictionResult(null);

    try {
      const data = await predictImage(fileOrBlob, filename);
      if (data && data.success) {
        setPredictionResult(data);
      } else {
        setToast({
          message: data?.error || 'Đã xảy ra lỗi khi phân tích ảnh.',
          type: 'error',
        });
      }
    } catch (err) {
      console.error('Prediction request error:', err);
      const errorMsg =
        err.response?.data?.error ||
        (err.code === 'ECONNABORTED'
          ? 'Quá thời gian chờ phản hồi từ máy chủ.'
          : 'Mất kết nối tới máy chủ. Vui lòng kiểm tra lại dịch vụ Backend.');
      setToast({ message: errorMsg, type: 'error' });
    } finally {
      setIsLoadingPrediction(false);
    }
  };

  // Khi người dùng chọn/kéo thả file ảnh từ máy tính
  const handleImageSelected = (file) => {
    setSelectedSampleId(null);
    setSelectedFile(file);

    const objectUrl = URL.createObjectURL(file);
    updatePreviewUrl(objectUrl);

    // Tự động kích hoạt suy luận ngay khi chọn ảnh
    executePrediction(file, file.name);
  };

  // Khi người dùng bấm vào một ảnh mẫu
  const handleSelectSample = async (sample) => {
    setSelectedSampleId(sample.id);

    const apiUrl = import.meta.env.VITE_API_URL || 'http://localhost:8000/api/v1';
    const serverBase = apiUrl.replace(/\/api\/v1\/?$/, '');
    const sampleFullUrl = `${serverBase}${sample.url}`;

    // Hiển thị ngay ảnh mẫu lên khung preview
    updatePreviewUrl(sampleFullUrl, false);

    setIsLoadingPrediction(true);
    setPredictionResult(null);

    try {
      // Tải blob ảnh từ static files của backend
      const blob = await fetchImageBlob(sample.url);
      const filename = `${sample.id}_${sample.original_label}.jpeg`;
      setSelectedFile(new File([blob], filename, { type: 'image/jpeg' }));

      // Gửi blob tới /predict
      await executePrediction(blob, filename);
    } catch (err) {
      console.error('Failed to load sample blob:', err);
      setToast({
        message: 'Không thể tải dữ liệu ảnh mẫu từ máy chủ để phân tích.',
        type: 'error',
      });
      setIsLoadingPrediction(false);
    }
  };

  // Đặt lại trạng thái (Phân tích ảnh khác / Xóa ảnh)
  const handleReset = () => {
    updatePreviewUrl(null);
    setSelectedFile(null);
    setSelectedSampleId(null);
    setPredictionResult(null);
  };

  return (
    <div className="min-h-screen flex flex-col bg-slate-50 text-slate-800">
      {/* 1. Header trên cùng */}
      <Header health={health} isCheckingHealth={isCheckingHealth} />

      {/* 2. Nội dung chính */}
      <main className="grow max-w-7xl w-full mx-auto px-4 sm:px-6 lg:px-8 py-8 space-y-8">
        {/* Banner khuyến cáo cố định theo quy định y tế */}
        <DisclaimerBanner />

        {/* Thư viện ảnh mẫu */}
        <SampleGallery
          samples={samples}
          isLoadingSamples={isLoadingSamples}
          selectedSampleId={selectedSampleId}
          onSelectSample={handleSelectSample}
          isLoadingPrediction={isLoadingPrediction}
        />

        {/* Khung chức năng chính: Upload & Kết quả phân loại */}
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-8 items-start">
          {/* Cột trái: Vùng tải ảnh (5/12 cột trên màn lớn) */}
          <div className="lg:col-span-5 space-y-4">
            <ImageUpload
              selectedFile={selectedFile}
              previewUrl={previewUrl}
              onImageSelected={handleImageSelected}
              onClearImage={handleReset}
              isLoading={isLoadingPrediction}
              onError={(errMsg) => setToast({ message: errMsg, type: 'error' })}
            />
          </div>

          {/* Cột phải: Bảng kết quả phân loại (7/12 cột trên màn lớn) */}
          <div className="lg:col-span-7">
            <PredictionResult
              result={predictionResult}
              isLoading={isLoadingPrediction}
              onReset={handleReset}
            />
          </div>
        </div>
      </main>

      {/* 3. Footer */}
      <footer className="border-t border-slate-200 bg-white py-6 mt-auto">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 flex flex-col sm:flex-row items-center justify-between text-xs text-slate-500 space-y-2 sm:space-y-0">
          <div>
            Đồ án tốt nghiệp: Hệ thống sàng lọc viêm phổi từ ảnh X-quang ngực (YOLOv8-cls & CLAHE)
          </div>
          <div className="text-slate-400">
            Môi trường: Windows • Python 3.12 • Node 22 • React 19 • Tailwind CSS v4
          </div>
        </div>
      </footer>

      {/* 4. Toast thông báo lỗi */}
      <Toast
        message={toast?.message}
        type={toast?.type}
        onClose={() => setToast(null)}
      />
    </div>
  );
}
