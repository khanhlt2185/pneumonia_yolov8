import React, { useState, useEffect, useRef } from 'react';
import Header from './components/Header';
import DisclaimerBanner from './components/DisclaimerBanner';
import XrayStudio from './components/XrayStudio';
import ClinicalReport from './components/ClinicalReport';
import SessionHistory from './components/SessionHistory';
import Toast from './components/Toast';
import { fetchHealth, fetchSamples, fetchImageBlob, predictImage } from './services/api';

export default function App() {
  // 1. Quản lý Theme Light / Dark
  const [isDark, setIsDark] = useState(() => {
    const saved = localStorage.getItem('theme');
    if (saved) return saved === 'dark';
    return window.matchMedia('(prefers-color-scheme: dark)').matches;
  });

  useEffect(() => {
    if (isDark) {
      document.documentElement.classList.add('dark');
      localStorage.setItem('theme', 'dark');
    } else {
      document.documentElement.classList.remove('dark');
      localStorage.setItem('theme', 'light');
    }
  }, [isDark]);

  const toggleTheme = () => setIsDark((prev) => !prev);

  // 2. Trạng thái kết nối Server & Ảnh mẫu
  const [health, setHealth] = useState(null);
  const [isCheckingHealth, setIsCheckingHealth] = useState(true);
  const [samples, setSamples] = useState([]);
  const [isLoadingSamples, setIsLoadingSamples] = useState(true);

  // 3. Trạng thái ảnh đang thao tác
  const [selectedFile, setSelectedFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState(null);
  const [selectedSampleId, setSelectedSampleId] = useState(null);

  // 4. Trạng thái kết quả & Lịch sử phiên (chỉ trên RAM)
  const [isLoadingPrediction, setIsLoadingPrediction] = useState(false);
  const [predictionResult, setPredictionResult] = useState(null);
  const [sessionHistory, setSessionHistory] = useState([]);
  const [toast, setToast] = useState(null);

  // Quản lý Blob URLs trong bộ nhớ để thu hồi tránh rò rỉ RAM
  const blobUrlsRef = useRef(new Set());

  const registerBlobUrl = (url) => {
    if (url && url.startsWith('blob:')) {
      blobUrlsRef.current.add(url);
    }
  };

  useEffect(() => {
    const currentBlobs = blobUrlsRef.current;
    return () => {
      currentBlobs.forEach((url) => URL.revokeObjectURL(url));
    };
  }, []);

  // Khởi tạo ứng dụng: kiểm tra máy chủ và nạp ảnh mẫu
  useEffect(() => {
    const initApp = async () => {
      // Health check
      try {
        const healthData = await fetchHealth();
        setHealth(healthData);
      } catch (err) {
        console.error('Health check failed:', err);
        setToast({
          message: 'Không thể kết nối máy chủ backend (cổng 8000). Vui lòng đảm bảo server đã chạy.',
          type: 'error',
        });
      } finally {
        setIsCheckingHealth(false);
      }

      // Nạp ảnh mẫu
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

  // Thực thi suy luận qua API
  const executePrediction = async (fileOrBlob, filename, currentUrl) => {
    setIsLoadingPrediction(true);
    setPredictionResult(null);

    try {
      const data = await predictImage(fileOrBlob, filename);
      if (data && data.success) {
        setPredictionResult(data);

        // Lưu vào lịch sử phiên làm việc (tối đa 8 ca gần nhất trong RAM)
        const historyItem = {
          id: Date.now(),
          previewUrl: currentUrl,
          fileName: filename,
          timestamp: new Date().toLocaleTimeString('vi-VN', {
            hour: '2-digit',
            minute: '2-digit',
          }),
          predictionResult: data,
        };

        setSessionHistory((prev) => [
          historyItem,
          ...prev.filter((item) => item.fileName !== filename),
        ].slice(0, 8));
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
          : 'Mất kết nối tới máy chủ. Vui lòng kiểm tra lại dịch vụ backend.');
      setToast({ message: errorMsg, type: 'error' });
    } finally {
      setIsLoadingPrediction(false);
    }
  };

  // Chọn ảnh từ máy tính (Upload hoặc kéo thả)
  const handleImageSelected = (file) => {
    setSelectedSampleId(null);
    setSelectedFile(file);

    const objectUrl = URL.createObjectURL(file);
    registerBlobUrl(objectUrl);
    setPreviewUrl(objectUrl);

    executePrediction(file, file.name, objectUrl);
  };

  // Chọn ảnh mẫu từ danh sách
  const handleSelectSample = async (sample) => {
    setSelectedSampleId(sample.id);

    const apiUrl = import.meta.env.VITE_API_URL || 'http://localhost:8000/api/v1';
    const serverBase = apiUrl.replace(/\/api\/v1\/?$/, '');
    const sampleFullUrl = `${serverBase}${sample.url}`;

    setPreviewUrl(sampleFullUrl);
    setIsLoadingPrediction(true);
    setPredictionResult(null);

    try {
      const blob = await fetchImageBlob(sample.url);
      const filename = `${sample.id}_${sample.original_label}.jpeg`;
      setSelectedFile(new File([blob], filename, { type: 'image/jpeg' }));

      await executePrediction(blob, filename, sampleFullUrl);
    } catch (err) {
      console.error('Failed to load sample blob:', err);
      setToast({
        message: 'Không thể tải dữ liệu ảnh mẫu từ máy chủ để phân tích.',
        type: 'error',
      });
      setIsLoadingPrediction(false);
    }
  };

  // Chọn xem lại một ca trong lịch sử phiên
  const handleSelectHistoryItem = (item) => {
    setPreviewUrl(item.previewUrl);
    setSelectedFile(new File([], item.fileName));
    setSelectedSampleId(null);
    setPredictionResult(item.predictionResult);
  };

  // Đặt lại trạng thái (Phân tích ảnh khác)
  const handleReset = () => {
    setPreviewUrl(null);
    setSelectedFile(null);
    setSelectedSampleId(null);
    setPredictionResult(null);
  };

  // Xóa lịch sử phiên trong RAM
  const handleClearHistory = () => {
    setSessionHistory([]);
  };

  return (
    <div className="min-h-screen flex flex-col bg-slate-50 dark:bg-[#0B0F17] text-slate-800 dark:text-slate-100 transition-colors">
      {/* 1. Header trên cùng */}
      <Header
        health={health}
        isCheckingHealth={isCheckingHealth}
        isDark={isDark}
        onToggleTheme={toggleTheme}
      />

      {/* 2. Không gian làm việc chính */}
      <main className="grow max-w-7xl w-full mx-auto px-4 sm:px-6 lg:px-8 py-5 sm:py-6 space-y-5">
        {/* Banner khuyến cáo y khoa rút gọn bắt buộc */}
        <DisclaimerBanner />

        {/* Khung chức năng: Studio X-quang (Cột trái) & Bảng kết quả (Cột phải) */}
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-5 items-start">
          {/* Cột trái: Studio X-quang, Viewer, Điều khiển Grad-CAM, Dải ảnh mẫu (7/12 cột) */}
          <div className="lg:col-span-7">
            <XrayStudio
              selectedFile={selectedFile}
              previewUrl={previewUrl}
              predictionResult={predictionResult}
              onImageSelected={handleImageSelected}
              onClearImage={handleReset}
              isLoadingPrediction={isLoadingPrediction}
              onError={(errMsg) => setToast({ message: errMsg, type: 'error' })}
              samples={samples}
              isLoadingSamples={isLoadingSamples}
              selectedSampleId={selectedSampleId}
              onSelectSample={handleSelectSample}
            />
          </div>

          {/* Cột phải: Bảng kết quả phân loại lâm sàng (5/12 cột) */}
          <div className="lg:col-span-5">
            <ClinicalReport
              result={predictionResult}
              isLoading={isLoadingPrediction}
              onReset={handleReset}
            />
          </div>
        </div>

        {/* 3. Khay lịch sử phiên làm việc trong RAM */}
        <SessionHistory
          history={sessionHistory}
          currentResult={predictionResult}
          onSelectHistoryItem={handleSelectHistoryItem}
          onClearHistory={handleClearHistory}
        />
      </main>

      {/* 4. Thông báo Toast */}
      <Toast
        message={toast?.message}
        type={toast?.type}
        onClose={() => setToast(null)}
      />
    </div>
  );
}
