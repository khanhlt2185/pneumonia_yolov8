import React from 'react';
import {
  AlertTriangle,
  ShieldCheck,
  RotateCcw,
  FileText,
  Activity,
} from 'lucide-react';

export default function ClinicalReport({ result, isLoading, onReset }) {
  // Trạng thái đang tải suy luận
  if (isLoading) {
    return (
      <div className="bg-white dark:bg-slate-900 rounded-2xl border border-slate-200 dark:border-slate-800 shadow-xs p-6 sm:p-8 flex flex-col items-center justify-center min-h-[420px] space-y-5 transition-colors">
        <div className="relative">
          <div className="w-14 h-14 rounded-full border-3 border-teal-100 dark:border-teal-900/40 border-t-teal-600 dark:border-t-teal-400 animate-spin" />
          <div className="absolute inset-0 flex items-center justify-center">
            <Activity className="h-5 w-5 text-teal-600 dark:text-teal-400 animate-pulse" />
          </div>
        </div>
        <div className="text-center space-y-1">
          <h3 className="text-sm font-bold text-slate-800 dark:text-slate-200">
            Đang phân tích hình ảnh X-quang...
          </h3>
          <p className="text-xs text-slate-400 dark:text-slate-500 max-w-xs">
            Hệ thống đang áp dụng tiền xử lý và trích xuất đặc trưng
          </p>
        </div>
        <div className="w-full max-w-xs space-y-2 pt-2">
          <div className="h-3.5 bg-slate-100 dark:bg-slate-800 rounded-full animate-shimmer" />
          <div className="h-8 bg-slate-100 dark:bg-slate-800 rounded-lg animate-shimmer" />
        </div>
      </div>
    );
  }

  // Trạng thái chưa nạp ảnh
  if (!result || !result.success) {
    return (
      <div className="bg-white dark:bg-slate-900 rounded-2xl border border-dashed border-slate-300 dark:border-slate-700/80 p-8 sm:p-12 flex flex-col items-center justify-center min-h-[420px] text-center transition-colors">
        <div className="w-12 h-12 rounded-2xl bg-slate-50 dark:bg-slate-800/60 border border-slate-200 dark:border-slate-700 flex items-center justify-center text-slate-400 dark:text-slate-500 mb-3.5">
          <FileText className="h-6 w-6 stroke-[1.5]" />
        </div>
        <h3 className="text-sm font-bold text-slate-700 dark:text-slate-300">
          Chưa có kết quả phân loại
        </h3>
        <p className="text-xs text-slate-400 dark:text-slate-500 max-w-xs mt-1 leading-relaxed">
          Vui lòng tải lên một ảnh X-quang ngực hoặc chọn ảnh mẫu kiểm thử ở cột bên trái để thực hiện sàng lọc.
        </p>
      </div>
    );
  }

  const isPneumonia = result.prediction === 'PNEUMONIA';
  const probNormal = Math.round((result.probabilities?.NORMAL || 0) * 100);
  const probPneumonia = Math.round((result.probabilities?.PNEUMONIA || 0) * 100);

  return (
    <div className="bg-white dark:bg-slate-900 rounded-2xl border border-slate-200 dark:border-slate-800 shadow-xs p-6 sm:p-7 space-y-5 transition-colors">
      {/* 1. Tiêu đề bảng kết quả & nút Reset */}
      <div className="flex items-center justify-between border-b border-slate-100 dark:border-slate-800 pb-3.5">
        <div>
          <span className="text-[11px] font-bold text-slate-400 dark:text-slate-500 tracking-wider uppercase">
            Đánh giá sơ bộ từ mô hình
          </span>
          <h2 className="text-base font-bold text-slate-900 dark:text-slate-100 mt-0.5">
            Kết quả sàng lọc X-quang
          </h2>
        </div>
        <button
          id="btn-reset-prediction"
          type="button"
          onClick={onReset}
          className="inline-flex items-center space-x-1.5 px-3 py-1.5 text-xs font-semibold text-slate-700 dark:text-slate-300 bg-slate-100 dark:bg-slate-800 hover:bg-slate-200 dark:hover:bg-slate-700 rounded-lg transition-colors cursor-pointer border border-slate-200 dark:border-slate-700"
        >
          <RotateCcw className="h-3.5 w-3.5" />
          <span>Phân tích ảnh khác</span>
        </button>
      </div>

      {/* 2. Card trạng thái phân loại nổi bật chuẩn lâm sàng */}
      <div
        className={`rounded-2xl p-5 border transition-all ${
          isPneumonia
            ? 'bg-rose-50/70 dark:bg-rose-950/40 border-rose-200 dark:border-rose-900/60 ring-2 ring-rose-100 dark:ring-rose-950'
            : 'bg-emerald-50/70 dark:bg-emerald-950/40 border-emerald-200 dark:border-emerald-900/60 ring-2 ring-emerald-100 dark:ring-emerald-950'
        }`}
      >
        <div className="flex items-start space-x-3.5">
          <div
            className={`p-2.5 rounded-xl shrink-0 ${
              isPneumonia
                ? 'bg-rose-600 text-white'
                : 'bg-emerald-600 text-white'
            }`}
          >
            {isPneumonia ? (
              <AlertTriangle className="h-6 w-6 stroke-[2.2]" />
            ) : (
              <ShieldCheck className="h-6 w-6 stroke-[2.2]" />
            )}
          </div>
          <div className="space-y-1 grow">
            <span
              className={`inline-block px-2 py-0.5 rounded text-[10px] font-bold uppercase tracking-wider ${
                isPneumonia
                  ? 'bg-rose-100 dark:bg-rose-900/60 text-rose-800 dark:text-rose-200'
                  : 'bg-emerald-100 dark:bg-emerald-900/60 text-emerald-800 dark:text-emerald-200'
              }`}
            >
              {result.prediction}
            </span>
            <h3
              className={`text-lg sm:text-xl font-extrabold tracking-tight ${
                isPneumonia
                  ? 'text-rose-950 dark:text-rose-100'
                  : 'text-emerald-950 dark:text-emerald-100'
              }`}
            >
              {isPneumonia ? 'Nghi ngờ viêm phổi' : 'Chưa ghi nhận dấu hiệu nghi ngờ'}
            </h3>
            {!isPneumonia && (
              <p className="text-xs text-emerald-800 dark:text-emerald-300 font-medium">
                * Lưu ý: Kết quả này không loại trừ bệnh.
              </p>
            )}
          </div>
        </div>
      </div>

      {/* 3. Phân bổ xác suất dự đoán (NORMAL vs PNEUMONIA) */}
      <div className="space-y-3 bg-slate-50 dark:bg-slate-800/50 rounded-xl p-4 border border-slate-200/80 dark:border-slate-800">
        <div className="flex items-center justify-between text-xs font-semibold text-slate-700 dark:text-slate-300">
          <span>Phân bố xác suất mô hình:</span>
          <span className="font-mono text-slate-900 dark:text-slate-100">
            {isPneumonia ? `${probPneumonia}% PNEUMONIA` : `${probNormal}% NORMAL`}
          </span>
        </div>

        {/* Thanh tiến trình kép trực quan */}
        <div className="w-full bg-slate-200 dark:bg-slate-700 rounded-full h-3 overflow-hidden flex shadow-inner">
          <div
            className="h-full bg-emerald-600 transition-all duration-500"
            style={{ width: `${probNormal}%` }}
            title={`NORMAL: ${probNormal}%`}
          />
          <div
            className="h-full bg-rose-600 transition-all duration-500"
            style={{ width: `${probPneumonia}%` }}
            title={`PNEUMONIA: ${probPneumonia}%`}
          />
        </div>

        {/* Bảng số liệu chi tiết 2 lớp */}
        <div className="grid grid-cols-2 gap-2.5 pt-1 text-xs">
          <div className="bg-white dark:bg-slate-800 p-2.5 rounded-lg border border-slate-200 dark:border-slate-700 flex justify-between items-center">
            <span className="text-slate-500 dark:text-slate-400 font-medium">NORMAL:</span>
            <span className="font-mono font-bold text-slate-800 dark:text-slate-200">
              {probNormal}%
            </span>
          </div>
          <div className="bg-white dark:bg-slate-800 p-2.5 rounded-lg border border-slate-200 dark:border-slate-700 flex justify-between items-center">
            <span className="text-slate-500 dark:text-slate-400 font-medium">PNEUMONIA:</span>
            <span className="font-mono font-bold text-slate-800 dark:text-slate-200">
              {probPneumonia}%
            </span>
          </div>
        </div>
      </div>

      {/* 4. Lời khuyến nghị y tế cố định từ API */}
      <div className="bg-teal-50/70 dark:bg-teal-950/40 border border-teal-200 dark:border-teal-900/60 rounded-xl p-4">
        <span className="text-[11px] font-bold text-teal-900 dark:text-teal-300 uppercase tracking-wider block mb-1">
          Lời khuyến nghị
        </span>
        <p className="text-xs sm:text-sm text-teal-950 dark:text-teal-100 font-medium leading-relaxed">
          {result.advice}
        </p>
      </div>
    </div>
  );
}
