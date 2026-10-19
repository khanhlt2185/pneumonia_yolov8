import React, { useRef, useState, useEffect } from 'react';
import { UploadCloud, FileImage, X, AlertTriangle, CheckCircle2 } from 'lucide-react';

const MAX_SIZE_MB = 10;
const MAX_BYTES = MAX_SIZE_MB * 1024 * 1024;
const ALLOWED_EXTENSIONS = ['image/jpeg', 'image/png', 'image/jpg'];

export default function ImageUpload({
  selectedFile,
  previewUrl,
  onImageSelected,
  onClearImage,
  isLoading,
  onError,
}) {
  const [isDragging, setIsDragging] = useState(false);
  const fileInputRef = useRef(null);

  // Xử lý validate file trước khi nhận
  const validateAndHandleFile = (file) => {
    if (!file) return;

    // Kiểm tra định dạng file
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

    // Kiểm tra dung lượng file
    if (file.size > MAX_BYTES) {
      onError(`Dung lượng ảnh (${(file.size / (1024 * 1024)).toFixed(1)} MB) vượt quá giới hạn tối đa ${MAX_SIZE_MB} MB.`);
      return;
    }

    onImageSelected(file);
  };

  const handleDragOver = (e) => {
    e.preventDefault();
    e.stopPropagation();
    if (!isLoading) setIsDragging(true);
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
    if (isLoading) return;

    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      const file = e.dataTransfer.files[0];
      validateAndHandleFile(file);
    }
  };

  const handleFileInputChange = (e) => {
    if (e.target.files && e.target.files.length > 0) {
      const file = e.target.files[0];
      validateAndHandleFile(file);
    }
  };

  return (
    <div className="bg-white rounded-2xl border border-slate-200 shadow-sm p-6">
      <div className="flex items-center justify-between mb-4">
        <div>
          <h2 className="text-base font-bold text-slate-900 flex items-center space-x-2">
            <span>Tải ảnh X-quang lồng ngực</span>
          </h2>
          <p className="text-xs text-slate-500">
            Hỗ trợ ảnh định dạng JPG, JPEG, PNG (Tối đa {MAX_SIZE_MB}MB)
          </p>
        </div>
        {selectedFile && !isLoading && (
          <button
            onClick={onClearImage}
            className="text-xs font-semibold text-rose-600 hover:text-rose-700 bg-rose-50 hover:bg-rose-100 px-2.5 py-1.5 rounded-lg transition-colors flex items-center space-x-1"
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
        disabled={isLoading}
      />

      {!previewUrl ? (
        /* Vùng kéo thả khi chưa có ảnh */
        <div
          id="dropzone-upload"
          onDragOver={handleDragOver}
          onDragLeave={handleDragLeave}
          onDrop={handleDrop}
          onClick={() => !isLoading && fileInputRef.current?.click()}
          className={`border-2 border-dashed rounded-xl p-8 sm:p-10 text-center cursor-pointer transition-all duration-200 ${
            isDragging
              ? 'border-sky-500 bg-sky-50/60 ring-4 ring-sky-100 scale-[1.01]'
              : 'border-slate-300 hover:border-sky-400 hover:bg-slate-50/70 bg-slate-50/30'
          } ${isLoading ? 'opacity-50 cursor-not-allowed' : ''}`}
        >
          <div className="flex flex-col items-center justify-center space-y-3">
            <div className={`p-4 rounded-2xl ${isDragging ? 'bg-sky-500 text-white shadow-lg' : 'bg-sky-50 text-sky-600'} transition-colors`}>
              <UploadCloud className="h-8 w-8 stroke-[2]" />
            </div>
            <div>
              <p className="text-sm font-semibold text-slate-800">
                Kéo thả file ảnh X-quang vào đây, hoặc{' '}
                <span id="btn-browse-file" className="text-sky-600 hover:underline">duyệt từ máy tính</span>
              </p>
              <p className="text-xs text-slate-400 mt-1">
                Ảnh sẽ được mã hóa an toàn trên RAM, không lưu đĩa cứng
              </p>
            </div>
            <div className="flex items-center space-x-2 text-[11px] text-slate-500 bg-slate-100/80 px-3 py-1 rounded-full">
              <span>Độ phân giải gợi ý: $224 \times 224$ trở lên</span>
            </div>
          </div>
        </div>
      ) : (
        /* Vùng xem trước ảnh đã chọn */
        <div className="relative rounded-xl border border-slate-200 overflow-hidden bg-slate-950 flex flex-col items-center group">
          <div className="w-full max-h-[380px] flex items-center justify-center p-3 relative">
            <img
              src={previewUrl}
              alt="Ảnh X-quang đã chọn"
              className="max-h-[350px] w-auto object-contain rounded-lg shadow-md"
            />
            {/* Overlay nhãn thông tin file */}
            <div className="absolute bottom-5 left-5 bg-slate-900/85 backdrop-blur-md text-white text-xs px-3 py-1.5 rounded-lg border border-slate-700/60 shadow flex items-center space-x-2">
              <FileImage className="h-4 w-4 text-sky-400" />
              <span className="font-medium truncate max-w-[200px] sm:max-w-xs">
                {selectedFile?.name || 'Ảnh X-quang'}
              </span>
              <span className="text-slate-400 text-[11px]">
                ({(selectedFile?.size / 1024).toFixed(0)} KB)
              </span>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
