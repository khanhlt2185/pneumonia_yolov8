import React from 'react';
import { Sparkles } from 'lucide-react';

export default function SampleGallery({
  samples,
  isLoadingSamples,
  selectedSampleId,
  onSelectSample,
  isLoadingPrediction,
}) {
  const getBackendBase = () => {
    const apiUrl = import.meta.env.VITE_API_URL || 'http://localhost:8000/api/v1';
    return apiUrl.replace(/\/api\/v1\/?$/, '');
  };

  const serverBase = getBackendBase();

  return (
    <div className="bg-white rounded-2xl border border-slate-200 shadow-sm p-6">
      <div className="flex items-center justify-between mb-4">
        <div>
          <h2 className="text-base font-bold text-slate-900 flex items-center space-x-2">
            <Sparkles className="h-4 w-4 text-sky-500" />
            <span>Ảnh mẫu kiểm thử trực tiếp</span>
          </h2>
          <p className="text-xs text-slate-500">
            Chọn một ảnh X-quang mẫu từ tập kiểm thử để phân tích nhanh
          </p>
        </div>
      </div>

      {isLoadingSamples ? (
        /* Skeleton loading */
        <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-6 gap-3">
          {[1, 2, 3, 4, 5, 6].map((i) => (
            <div key={i} className="aspect-square rounded-xl bg-slate-100 animate-shimmer" />
          ))}
        </div>
      ) : samples.length === 0 ? (
        <div className="text-center py-6 text-xs text-slate-400">
          Chưa nạp được ảnh mẫu từ máy chủ.
        </div>
      ) : (
        <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-6 gap-3">
          {samples.map((sample) => {
            const isSelected = selectedSampleId === sample.id;
            const fullImgUrl = `${serverBase}${sample.url}`;
            const isPneumonia = sample.original_label === 'PNEUMONIA';

            return (
              <button
                key={sample.id}
                id={`btn-sample-${sample.id}`}
                onClick={() => onSelectSample(sample)}
                disabled={isLoadingPrediction}
                className={`group relative flex flex-col items-center rounded-xl overflow-hidden border-2 transition-all duration-200 text-left ${
                  isSelected
                    ? 'border-sky-500 ring-4 ring-sky-100 scale-102 shadow-md'
                    : 'border-slate-200 hover:border-slate-300 hover:shadow-sm bg-slate-50'
                } ${isLoadingPrediction ? 'opacity-60 cursor-not-allowed' : 'cursor-pointer'}`}
              >
                <div className="w-full aspect-square bg-slate-900 overflow-hidden relative">
                  <img
                    src={fullImgUrl}
                    alt={`Ảnh mẫu ${sample.id}`}
                    className="w-full h-full object-cover transition-transform duration-300 group-hover:scale-105"
                    loading="lazy"
                  />
                  {/* Badge nhãn thực tế */}
                  <div className="absolute top-1.5 left-1.5">
                    <span
                      className={`px-1.5 py-0.5 rounded text-[10px] font-bold tracking-wider shadow-sm uppercase ${
                        isPneumonia
                          ? 'bg-rose-500/90 text-white'
                          : 'bg-emerald-600/90 text-white'
                      }`}
                    >
                      {sample.original_label}
                    </span>
                  </div>
                </div>
                <div className="w-full py-1.5 px-2 bg-white text-center">
                  <span className="text-[11px] font-semibold text-slate-700 block truncate">
                    {sample.id}
                  </span>
                </div>
              </button>
            );
          })}
        </div>
      )}
    </div>
  );
}
