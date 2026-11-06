import React, { useState, useRef } from 'react';
import {
  UploadCloud,
  FileImage,
  X,
  ZoomIn,
  ZoomOut,
  RotateCcw,
  Sparkles,
} from 'lucide-react';
import { WEAK_PEAK, REF_PEAK } from '../config/cam';

const MAX_SIZE_MB = 10;
const MAX_BYTES = MAX_SIZE_MB * 1024 * 1024;
const ALLOWED_EXTENSIONS = ['image/jpeg', 'image/png', 'image/jpg'];

export default function XrayStudio({
  selectedFile,
  previewUrl,
  explainResult,
  showHeatmap = false,
  heatmapOpacity = 50,
  onHeatmapOpacityChange,
  onImageSelected,
  onClearImage,
  isLoadingPrediction,
  onError,
  samples = [],
  isLoadingSamples = false,
  selectedSampleId,
  onSelectSample,
}) {
  const [isDragging, setIsDragging] = useState(false);
  const fileInputRef = useRef(null);

  // Trạng thái công cụ xem ảnh DICOM
  const [zoom, setZoom] = useState(1);
  const [pan, setPan] = useState({ x: 0, y: 0 });
  const [isInverted, setIsInverted] = useState(false);
  const [isPanning, setIsPanning] = useState(false);
  const [startPan, setStartPan] = useState({ x: 0, y: 0 });
  const [imageAspectRatio, setImageAspectRatio] = useState(null);

  // Tự động reset góc nhìn khi đổi ảnh theo React pattern chuẩn (không dùng setState trong effect)
  const [prevPreviewUrl, setPrevPreviewUrl] = useState(previewUrl);
  if (prevPreviewUrl !== previewUrl) {
    setPrevPreviewUrl(previewUrl);
    setZoom(1);
    setPan({ x: 0, y: 0 });
    setIsInverted(false);
    setImageAspectRatio(null);
  }

  const handleImageLoad = (e) => {
    const { naturalWidth, naturalHeight } = e.target;
    if (naturalWidth && naturalHeight) {
      setImageAspectRatio(naturalWidth / naturalHeight);
    }
  };

  // Kiểm tra file đầu vào
  const validateAndHandleFile = (file) => {
    if (!file) return;

    const fileType = file.type.toLowerCase();
    const fileName = file.name.toLowerCase();
    const isValidExtension =
      ALLOWED_EXTENSIONS.includes(fileType) ||
      fileName.endsWith('.jpg') ||
      fileName.endsWith('.jpeg') ||
      fileName.endsWith('.png');

    if (!isValidExtension) {
      onError('Định dạng file không được hỗ trợ. Vui lòng chỉ tải ảnh JPG, JPEG hoặc PNG.');
      return;
    }

    if (file.size > MAX_BYTES) {
      onError(`Dung lượng ảnh (${(file.size / (1024 * 1024)).toFixed(1)} MB) vượt quá giới hạn tối đa ${MAX_SIZE_MB} MB.`);
      return;
    }

    onImageSelected(file);
  };

  const handleDragOver = (e) => {
    e.preventDefault();
    e.stopPropagation();
    if (!isLoadingPrediction) setIsDragging(true);
  };

  const handleDragLeave = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(false);
  };

  const handleDrop = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(false);
    if (isLoadingPrediction) return;

    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      validateAndHandleFile(e.dataTransfer.files[0]);
    }
  };

  const handleFileInputChange = (e) => {
    if (e.target.files && e.target.files.length > 0) {
      validateAndHandleFile(e.target.files[0]);
    }
  };

  // Điều khiển Zoom
  const handleZoomIn = () => setZoom((prev) => Math.min(prev + 0.25, 3.5));
  const handleZoomOut = () => {
    setZoom((prev) => {
      const next = Math.max(prev - 0.25, 1);
      if (next === 1) setPan({ x: 0, y: 0 });
      return next;
    });
  };

  const handleResetView = () => {
    setZoom(1);
    setPan({ x: 0, y: 0 });
    setIsInverted(false);
  };

  // Xử lý kéo ảnh (Pan) khi zoom > 1
  const handleMouseDown = (e) => {
    if (zoom <= 1) return;
    setIsPanning(true);
    setStartPan({ x: e.clientX - pan.x, y: e.clientY - pan.y });
  };

  const handleMouseMove = (e) => {
    if (!isPanning || zoom <= 1) return;
    setPan({
      x: e.clientX - startPan.x,
      y: e.clientY - startPan.y,
    });
  };

  const handleMouseUp = () => setIsPanning(false);

  // Xử lý dữ liệu giải thích Grad-CAM
  const rawBase64 = explainResult?.heatmap_base64;
  const heatmapSrc = rawBase64
    ? (rawBase64.startsWith('data:') ? rawBase64 : `data:image/png;base64,${rawBase64}`)
    : null;

  let x0 = 0, y0 = 0, x1 = 0, y1 = 0;
  if (Array.isArray(explainResult?.crop_box) && explainResult.crop_box.length >= 4) {
    [x0, y0, x1, y1] = explainResult.crop_box;
  } else if (explainResult?.crop_box) {
    x0 = explainResult.crop_box.x0 ?? explainResult.crop_box.left ?? 0;
    y0 = explainResult.crop_box.y0 ?? explainResult.crop_box.top ?? 0;
    x1 = explainResult.crop_box.x1 ?? explainResult.crop_box.right ?? 0;
    y1 = explainResult.crop_box.y1 ?? explainResult.crop_box.bottom ?? 0;
  }

  let imgWidth = 0, imgHeight = 0;
  if (Array.isArray(explainResult?.image_size) && explainResult.image_size.length >= 2) {
    [imgWidth, imgHeight] = explainResult.image_size;
  } else if (explainResult?.image_size) {
    imgWidth = explainResult.image_size.width ?? 0;
    imgHeight = explainResult.image_size.height ?? 0;
  }

  const leftPercent = imgWidth > 0 ? (x0 / imgWidth) * 100 : 0;
  const topPercent = imgHeight > 0 ? (y0 / imgHeight) * 100 : 0;
  const widthPercent = imgWidth > 0 ? ((x1 - x0) / imgWidth) * 100 : 100;
  const heightPercent = imgHeight > 0 ? ((y1 - y0) / imgHeight) * 100 : 100;

  const rawPeak = explainResult?.raw_peak ?? 0;
  const peakRatio = REF_PEAK > 0 ? Math.min(1, Math.max(0, rawPeak / REF_PEAK)) : 1;
  const effectiveOpacity = (heatmapOpacity / 100) * peakRatio;

  const isWeakEvidence = Boolean(
    explainResult?.is_empty || (explainResult?.raw_peak !== undefined && explainResult.raw_peak < WEAK_PEAK)
  );

  return (
    <div className="bg-white dark:bg-slate-900 rounded-2xl border border-slate-200 dark:border-slate-800 shadow-xs p-5 space-y-4 transition-colors">
      {/* 1. Thanh tiêu đề khung soi ảnh */}
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-base font-bold text-slate-900 dark:text-slate-100 flex items-center space-x-2">
            <span>Studio Soi Ảnh X-Quang</span>
          </h2>
          <p className="text-xs text-slate-500 dark:text-slate-400">
            Hỗ trợ kéo thả ảnh JPG/PNG hoặc chọn ảnh mẫu kiểm thử
          </p>
        </div>

        {selectedFile && !isLoadingPrediction && (
          <button
            type="button"
            onClick={onClearImage}
            className="text-xs font-semibold text-rose-600 dark:text-rose-400 hover:text-rose-700 bg-rose-50 dark:bg-rose-950/50 hover:bg-rose-100 dark:hover:bg-rose-900/60 px-2.5 py-1.5 rounded-lg transition-colors flex items-center space-x-1 cursor-pointer"
          >
            <X className="h-3.5 w-3.5" />
            <span>Xóa ảnh</span>
          </button>
        )}
      </div>

      <input
        type="file"
        id="file-input-xray"
        ref={fileInputRef}
        onChange={handleFileInputChange}
        accept=".jpg,.jpeg,.png,image/jpeg,image/png"
        className="hidden"
        disabled={isLoadingPrediction}
      />

      {/* 2. Vùng hiển thị chính: Upload Dropzone hoặc DICOM Viewer */}
      {!previewUrl ? (
        <div
          id="dropzone-upload"
          onDragOver={handleDragOver}
          onDragLeave={handleDragLeave}
          onDrop={handleDrop}
          onClick={() => !isLoadingPrediction && fileInputRef.current?.click()}
          className={`border-2 border-dashed rounded-xl p-8 sm:p-12 text-center cursor-pointer transition-all duration-200 ${
            isDragging
              ? 'border-teal-500 bg-teal-50/50 dark:bg-teal-950/30 scale-[1.01]'
              : 'border-slate-300 dark:border-slate-700 hover:border-teal-500 dark:hover:border-teal-400 hover:bg-slate-50/70 dark:hover:bg-slate-800/40 bg-slate-50/40 dark:bg-slate-800/20'
          } ${isLoadingPrediction ? 'opacity-50 cursor-not-allowed' : ''}`}
        >
          <div className="flex flex-col items-center justify-center space-y-3">
            <div className={`p-4 rounded-2xl ${isDragging ? 'bg-teal-600 text-white shadow-md' : 'bg-teal-50 dark:bg-teal-950/80 text-teal-600 dark:text-teal-400'} transition-colors`}>
              <UploadCloud className="h-8 w-8 stroke-[2]" />
            </div>
            <div>
              <p className="text-sm font-semibold text-slate-800 dark:text-slate-200">
                Kéo thả file ảnh X-quang vào đây, hoặc{' '}
                <span className="text-teal-600 dark:text-teal-400 hover:underline">duyệt từ máy</span>
              </p>
              <p className="text-xs text-slate-400 dark:text-slate-500 mt-1">
                Tối đa 10 MB • Xử lý bảo mật hoàn toàn trên RAM
              </p>
            </div>
          </div>
        </div>
      ) : (
        <div className="space-y-3">
          {/* Khung xem phim X-quang nền tối trung tính */}
          <div
            className="relative rounded-xl border border-slate-800 overflow-hidden bg-[#090D14] h-[380px] sm:h-[420px] flex items-center justify-center select-none cursor-grab active:cursor-grabbing"
            onMouseDown={handleMouseDown}
            onMouseMove={handleMouseMove}
            onMouseUp={handleMouseUp}
            onMouseLeave={handleMouseUp}
          >
            {/* Lớp hiển thị ảnh với Zoom, Pan */}
            <div
              className="relative transition-transform duration-75 ease-out max-h-full max-w-full flex items-center justify-center"
              style={{
                transform: `scale(${zoom}) translate(${pan.x / zoom}px, ${pan.y / zoom}px)`,
              }}
            >
              {/* Khung bọc ôm sát kích thước hiển thị thật của ảnh X-quang */}
              <div
                className="relative inline-block max-h-[360px] sm:max-h-[400px] max-w-full"
                style={imageAspectRatio ? { aspectRatio: `${imageAspectRatio}` } : undefined}
              >
                {/* Lớp 1: Ảnh X-quang gốc */}
                <img
                  src={previewUrl}
                  alt="Ảnh X-quang lồng ngực"
                  draggable={false}
                  onLoad={handleImageLoad}
                  className="max-h-[360px] sm:max-h-[400px] w-auto h-auto max-w-full block object-contain rounded transition-all pointer-events-none"
                  style={{
                    filter: isInverted ? 'invert(1) contrast(1.15)' : 'none',
                  }}
                />

                {/* Lớp 2: Lớp phủ Grad-CAM và viền nét đứt Vùng mô hình xét */}
                {showHeatmap && explainResult && (
                  <div
                    className="absolute border-2 border-dashed border-amber-400 pointer-events-none rounded-xs overflow-hidden"
                    style={{
                      left: `${leftPercent}%`,
                      top: `${topPercent}%`,
                      width: `${widthPercent}%`,
                      height: `${heightPercent}%`,
                    }}
                  >
                    <span className="absolute top-1 left-1 bg-slate-900/90 text-amber-300 text-[10px] px-1.5 py-0.5 rounded font-medium border border-amber-400/40 z-10 pointer-events-none select-none">
                      Vùng mô hình xét
                    </span>

                    {!isWeakEvidence && heatmapSrc ? (
                      <img
                        src={heatmapSrc}
                        alt=""
                        aria-hidden="true"
                        className="w-full h-full object-fill pointer-events-none"
                        style={{
                          opacity: effectiveOpacity,
                        }}
                      />
                    ) : (
                      <div className="w-full h-full flex items-center justify-center p-2">
                        <span className="text-xs font-medium text-amber-200 text-center select-none">
                          Bằng chứng yếu: không hiển thị bản đồ.
                        </span>
                      </div>
                    )}
                  </div>
                )}
              </div>
            </div>

            {/* Thông tin tên file nhỏ góc dưới */}
            <div className="absolute bottom-3 left-3 bg-slate-900/85 backdrop-blur-xs text-slate-300 text-xs px-2.5 py-1 rounded-md border border-slate-700/60 shadow-xs flex items-center space-x-1.5 pointer-events-none">
              <FileImage className="h-3.5 w-3.5 text-teal-400" />
              <span className="font-medium truncate max-w-[160px] sm:max-w-xs">
                {selectedFile?.name || 'Ảnh X-quang'}
              </span>
            </div>

            {/* Chỉ báo Zoom khi đang phóng to */}
            {zoom > 1 && (
              <div className="absolute top-3 left-3 bg-slate-900/85 backdrop-blur-xs text-teal-400 text-xs px-2 py-0.5 rounded border border-slate-700 font-mono">
                {Math.round(zoom * 100)}%
              </div>
            )}
          </div>

          {/* 3. Thanh công cụ tương tác lâm sàng */}
          <div className="flex flex-wrap items-center justify-between gap-2 p-2.5 rounded-xl bg-slate-100 dark:bg-slate-800/80 border border-slate-200 dark:border-slate-700/80 text-xs">
            {/* Nhóm công cụ Zoom / Invert / Reset */}
            <div className="flex items-center space-x-1.5">
              <button
                type="button"
                onClick={handleZoomIn}
                disabled={zoom >= 3.5}
                title="Phóng to"
                className="p-1.5 rounded-lg bg-white dark:bg-slate-700 text-slate-700 dark:text-slate-200 hover:bg-slate-200 dark:hover:bg-slate-600 disabled:opacity-40 transition-colors cursor-pointer border border-slate-200 dark:border-slate-600"
              >
                <ZoomIn className="h-4 w-4" />
              </button>
              <button
                type="button"
                onClick={handleZoomOut}
                disabled={zoom <= 1}
                title="Thu nhỏ"
                className="p-1.5 rounded-lg bg-white dark:bg-slate-700 text-slate-700 dark:text-slate-200 hover:bg-slate-200 dark:hover:bg-slate-600 disabled:opacity-40 transition-colors cursor-pointer border border-slate-200 dark:border-slate-600"
              >
                <ZoomOut className="h-4 w-4" />
              </button>
              <button
                type="button"
                onClick={() => setIsInverted((prev) => !prev)}
                title="Đảo âm bản để tăng tương phản"
                className={`px-2 py-1.5 rounded-lg font-medium transition-colors cursor-pointer border ${
                  isInverted
                    ? 'bg-teal-600 text-white border-teal-600'
                    : 'bg-white dark:bg-slate-700 text-slate-700 dark:text-slate-200 border-slate-200 dark:border-slate-600 hover:bg-slate-200 dark:hover:bg-slate-600'
                }`}
              >
                Âm bản
              </button>
              <button
                type="button"
                onClick={handleResetView}
                title="Đặt lại góc nhìn"
                className="p-1.5 rounded-lg bg-white dark:bg-slate-700 text-slate-700 dark:text-slate-200 hover:bg-slate-200 dark:hover:bg-slate-600 transition-colors cursor-pointer border border-slate-200 dark:border-slate-600"
              >
                <RotateCcw className="h-4 w-4" />
              </button>
            </div>

            {/* Nhóm điều khiển độ đậm vùng mô hình chú ý */}
            <div className="flex items-center space-x-2">
              <label
                htmlFor="cam-opacity-slider"
                className="text-[11px] text-slate-600 dark:text-slate-300 font-medium whitespace-nowrap"
              >
                Độ đậm:
              </label>
              <div className="flex items-center space-x-1.5 pl-1">
                <span className="text-[11px] text-slate-500 dark:text-slate-400 font-mono w-7 text-right">
                  {heatmapOpacity}%
                </span>
                <input
                  id="cam-opacity-slider"
                  type="range"
                  min="0"
                  max="100"
                  value={heatmapOpacity}
                  onChange={(e) => onHeatmapOpacityChange && onHeatmapOpacityChange(Number(e.target.value))}
                  disabled={!showHeatmap || !explainResult}
                  aria-label="Độ đậm vùng mô hình chú ý"
                  className="w-18 sm:w-24 h-1.5 bg-slate-200 dark:bg-slate-600 rounded-lg appearance-none cursor-pointer accent-teal-600 disabled:opacity-40"
                  title="Độ đậm vùng mô hình chú ý"
                />
              </div>
            </div>
          </div>

          {/* Cảnh báo y đức & Chú giải màu cho vùng mô hình chú ý */}
          {showHeatmap && explainResult && (
            <div className="p-3 rounded-xl bg-slate-100 dark:bg-slate-800/90 border border-amber-300/60 dark:border-amber-700/60 text-xs space-y-2">
              <p className="text-slate-700 dark:text-slate-300 leading-relaxed font-medium">
                <span className="font-bold text-amber-800 dark:text-amber-400">Vùng mô hình chú ý:</span> chỉ cho biết vùng mô hình dùng khi phân tích, không phải vị trí tổn thương hay kết luận lâm sàng. Độ trung thực của bản đồ này đã được kiểm chứng là thấp.
              </p>
              <div className="flex items-center justify-between text-[11px] text-slate-500 dark:text-slate-400 pt-1 border-t border-slate-200 dark:border-slate-700/80">
                <span className="font-medium">Mức chú ý: thấp đến cao</span>
                <div className="flex items-center space-x-1.5">
                  <span className="text-[10px]">Thấp</span>
                  <div
                    className="w-24 h-2 rounded-full shadow-inner"
                    style={{
                      background:
                        'linear-gradient(to right, #30123b, #4662d7, #36aaf9, #1ae4b6, #72fe5e, #c7ef34, #fabb2a, #f56918, #cc2403, #7a0403)',
                    }}
                  />
                  <span className="text-[10px]">Cao</span>
                </div>
              </div>
            </div>
          )}
        </div>
      )}

      {/* 4. Dải ảnh mẫu kiểm thử thu gọn */}
      <div className="pt-2 border-t border-slate-100 dark:border-slate-800">
        <div className="flex items-center justify-between mb-2">
          <span className="text-xs font-semibold text-slate-600 dark:text-slate-300 flex items-center space-x-1.5">
            <Sparkles className="h-3.5 w-3.5 text-teal-500" />
            <span>Ảnh mẫu kiểm thử trực tiếp</span>
          </span>
          <span className="text-[11px] text-slate-400">Chọn ảnh để phân tích nhanh</span>
        </div>

        {isLoadingSamples ? (
          <div className="grid grid-cols-4 sm:grid-cols-6 gap-2">
            {[1, 2, 3, 4, 5, 6].map((i) => (
              <div key={i} className="aspect-square rounded-lg bg-slate-100 dark:bg-slate-800 animate-shimmer" />
            ))}
          </div>
        ) : samples.length === 0 ? (
          <div className="text-center py-3 text-xs text-slate-400">
            Chưa có ảnh mẫu từ máy chủ.
          </div>
        ) : (
          <div className="grid grid-cols-4 sm:grid-cols-6 gap-2">
            {samples.map((sample) => {
              const isSelected = selectedSampleId === sample.id;
              const apiUrl = import.meta.env.VITE_API_URL || 'http://localhost:8000/api/v1';
              const serverBase = apiUrl.replace(/\/api\/v1\/?$/, '');
              const fullImgUrl = `${serverBase}${sample.url}`;
              const isPneumonia = sample.original_label === 'PNEUMONIA';

              return (
                <button
                  key={sample.id}
                  id={`btn-sample-${sample.id}`}
                  type="button"
                  onClick={() => onSelectSample(sample)}
                  disabled={isLoadingPrediction}
                  className={`group relative rounded-lg overflow-hidden border-2 transition-all duration-150 text-left cursor-pointer ${
                    isSelected
                      ? 'border-teal-500 ring-2 ring-teal-500/30'
                      : 'border-slate-200 dark:border-slate-700 hover:border-teal-400 dark:hover:border-teal-500'
                  } ${isLoadingPrediction ? 'opacity-50 cursor-not-allowed' : ''}`}
                >
                  <div className="aspect-square bg-slate-950 overflow-hidden relative">
                    <img
                      src={fullImgUrl}
                      alt={`Ảnh mẫu ${sample.id}`}
                      className="w-full h-full object-cover transition-transform duration-200 group-hover:scale-105"
                      loading="lazy"
                    />
                    <div className="absolute top-1 left-1">
                      <span
                        className={`px-1 py-0.2 rounded text-[9px] font-bold tracking-wider uppercase ${
                          isPneumonia
                            ? 'bg-rose-500 text-white'
                            : 'bg-emerald-600 text-white'
                        }`}
                      >
                        {sample.original_label === 'PNEUMONIA' ? 'PNEU' : 'NORM'}
                      </span>
                    </div>
                  </div>
                </button>
              );
            })}
          </div>
        )}
      </div>
    </div>
  );
}
