import React from 'react';
import {
  AlertTriangle,
  CheckCircle2,
  Clock,
  RotateCcw,
  Sparkles,
  HelpCircle,
  FileText,
} from 'lucide-react';

export default function PredictionResult({ result, isLoading, onReset }) {
  if (isLoading) {
    return (
      <div className="bg-white rounded-2xl border border-slate-200 shadow-sm p-6 sm:p-8 flex flex-col items-center justify-center min-h-[380px] space-y-6">
        <div className="relative">
          <div className="w-16 h-16 rounded-full border-4 border-sky-100 border-t-sky-500 animate-spin" />
          <div className="absolute inset-0 flex items-center justify-center">
            <Sparkles className="h-6 w-6 text-sky-500 animate-pulse" />
          </div>
        </div>
        <div className="text-center space-y-1">
          <h3 className="text-base font-bold text-slate-800">
            Đang tiền xử lý & phân tích ảnh...
          </h3>
          <p className="text-xs text-slate-500 max-w-xs">
            Hệ thống đang áp dụng cân bằng tương phản CLAHE và tính toán xác suất qua YOLOv8
          </p>
        </div>
        {/* Skeleton progress bars */}
        <div className="w-full max-w-md space-y-3 pt-2">
          <div className="h-4 bg-slate-100 rounded-full animate-shimmer" />
          <div className="h-10 bg-slate-100 rounded-xl animate-shimmer" />
        </div>
      </div>
    );
  }

  if (!result || !result.success) {
    return (
      <div className="bg-white rounded-2xl border border-dashed border-slate-200 p-8 sm:p-12 flex flex-col items-center justify-center min-h-[380px] text-center">
        <div className="w-14 h-14 rounded-2xl bg-slate-50 border border-slate-100 flex items-center justify-center text-slate-400 mb-4">
          <FileText className="h-7 w-7 stroke-[1.5]" />
        </div>
        <h3 className="text-base font-bold text-slate-700">Chưa có kết quả phân loại</h3>
        <p className="text-xs text-slate-400 max-w-xs mt-1.5 leading-relaxed">
          Vui lòng tải lên một ảnh X-quang hoặc chọn một ảnh mẫu thử nghiệm phía trên để xem kết quả phân tích.
        </p>
      </div>
    );
  }

  const isPneumonia = result.prediction === 'PNEUMONIA';
  const confidencePercent = (result.confidence * 100).toFixed(1);
  const probNormal = ((result.probabilities?.NORMAL || 0) * 100).toFixed(1);
  const probPneumonia = ((result.probabilities?.PNEUMONIA || 0) * 100).toFixed(1);

  return (
    <div className="bg-white rounded-2xl border border-slate-200 shadow-sm p-6 sm:p-8 space-y-6">
      <div className="flex items-center justify-between border-b border-slate-100 pb-4">
        <div>
          <span className="text-xs font-bold text-slate-400 tracking-wider uppercase">
            Bảng kết quả phân loại
          </span>
          <h2 className="text-lg font-bold text-slate-900 mt-0.5">
            Đánh giá sơ bộ từ mô hình
          </h2>
        </div>
        <button
          id="btn-reset-prediction"
          onClick={onReset}
          className="inline-flex items-center space-x-1.5 px-3 py-1.5 text-xs font-semibold text-slate-700 bg-slate-100 hover:bg-slate-200 rounded-lg transition-colors cursor-pointer"
        >
          <RotateCcw className="h-3.5 w-3.5" />
          <span>Phân tích ảnh khác</span>
        </button>
      </div>

      {/* Card trạng thái nổi bật */}
      <div
        className={`rounded-2xl p-5 sm:p-6 border transition-all ${
          isPneumonia
            ? 'bg-gradient-to-br from-rose-50 via-rose-50/50 to-white border-rose-200 ring-4 ring-rose-50'
            : 'bg-gradient-to-br from-emerald-50 via-emerald-50/50 to-white border-emerald-200 ring-4 ring-emerald-50'
        }`}
      >
        <div className="flex items-start space-x-4">
          <div
            className={`p-3 rounded-2xl shrink-0 shadow-sm ${
              isPneumonia ? 'bg-rose-500 text-white' : 'bg-emerald-600 text-white'
            }`}
          >
            {isPneumonia ? (
              <AlertTriangle className="h-7 w-7 stroke-[2.2]" />
            ) : (
              <CheckCircle2 className="h-7 w-7 stroke-[2.2]" />
            )}
          </div>
          <div className="space-y-1 grow">
            <span
              className={`inline-block px-2.5 py-0.5 rounded-full text-xs font-bold uppercase tracking-wider ${
                isPneumonia
                  ? 'bg-rose-100 text-rose-800 border border-rose-200'
                  : 'bg-emerald-100 text-emerald-800 border border-emerald-200'
              }`}
            >
              {result.prediction}
            </span>
            <h3
              className={`text-xl sm:text-2xl font-extrabold tracking-tight ${
                isPneumonia ? 'text-rose-900' : 'text-emerald-950'
              }`}
            >
              {isPneumonia
                ? 'Nghi ngờ viêm phổi'
                : 'Chưa ghi nhận dấu hiệu nghi ngờ'}
            </h3>
            {!isPneumonia && (
              <p className="text-xs text-emerald-800 font-medium">
                * Lưu ý: Kết quả này không loại trừ bệnh.
              </p>
            )}
          </div>
        </div>
      </div>

      {/* Thanh tiến trình Xác suất dự đoán */}
      <div className="space-y-3 bg-slate-50/80 rounded-xl p-4 border border-slate-200/80">
        <div className="flex items-center justify-between text-xs font-semibold text-slate-700">
          <span className="flex items-center space-x-1.5">
            <span>Xác suất dự đoán lớp chính ({result.prediction}):</span>
          </span>
          <span className="text-sm font-bold text-slate-900">
            {confidencePercent}%
          </span>
        </div>
        <div className="w-full bg-slate-200 rounded-full h-3.5 overflow-hidden p-0.5 shadow-inner">
          <div
            className={`h-full rounded-full transition-all duration-500 ${
              isPneumonia
                ? 'bg-gradient-to-r from-rose-500 to-red-600'
                : 'bg-gradient-to-r from-emerald-500 to-teal-600'
            }`}
            style={{ width: `${Math.max(Number(confidencePercent), 3)}%` }}
          />
        </div>

        {/* Chi tiết phân phối xác suất cả 2 lớp */}
        <div className="grid grid-cols-2 gap-3 pt-2 text-xs">
          <div className="bg-white p-2.5 rounded-lg border border-slate-200 flex justify-between items-center">
            <span className="text-slate-500 font-medium">NORMAL:</span>
            <span className="font-bold text-slate-800">{probNormal}%</span>
          </div>
          <div className="bg-white p-2.5 rounded-lg border border-slate-200 flex justify-between items-center">
            <span className="text-slate-500 font-medium">PNEUMONIA:</span>
            <span className="font-bold text-slate-800">{probPneumonia}%</span>
          </div>
        </div>
      </div>

      {/* Lời khuyến nghị y tế cố định theo SPEC */}
      <div className="bg-blue-50/70 border border-blue-200/80 rounded-xl p-4">
        <span className="text-xs font-bold text-blue-900 uppercase tracking-wider block mb-1">
          Lời khuyến nghị
        </span>
        <p className="text-xs sm:text-sm text-blue-950 font-medium leading-relaxed">
          {result.advice}
        </p>
      </div>

      {/* Footer thời gian xử lý */}
      <div className="flex items-center justify-between text-xs text-slate-500 pt-2 border-t border-slate-100">
        <div className="flex items-center space-x-1.5">
          <Clock className="h-3.5 w-3.5 text-slate-400" />
          <span>Thời gian xử lý tại máy chủ:</span>
        </div>
        <span className="font-semibold text-slate-700 bg-slate-100 px-2 py-0.5 rounded">
          {result.inference_time_ms} ms
        </span>
      </div>
    </div>
  );
}
