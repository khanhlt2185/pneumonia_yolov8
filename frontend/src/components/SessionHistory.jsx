import React from 'react';
import { History, Trash2, AlertTriangle, ShieldCheck } from 'lucide-react';

export default function SessionHistory({
  history = [],
  currentResult,
  onSelectHistoryItem,
  onClearHistory,
}) {
  if (history.length === 0) return null;

  return (
    <div className="bg-white dark:bg-slate-900 rounded-2xl border border-slate-200 dark:border-slate-800 shadow-xs p-4 sm:p-5 transition-colors">
      <div className="flex items-center justify-between mb-3">
        <div className="flex items-center space-x-2">
          <History className="h-4 w-4 text-teal-600 dark:text-teal-400" />
          <h3 className="text-xs sm:text-sm font-bold text-slate-800 dark:text-slate-200">
            Lịch sử phiên làm việc
          </h3>
          <span className="text-[11px] text-slate-400 font-mono">
            ({history.length}/8 ca trong RAM)
          </span>
        </div>

        <button
          type="button"
          onClick={onClearHistory}
          className="text-xs text-slate-400 hover:text-rose-600 dark:hover:text-rose-400 flex items-center space-x-1 cursor-pointer transition-colors"
          title="Xóa lịch sử phiên trong bộ nhớ"
        >
          <Trash2 className="h-3.5 w-3.5" />
          <span className="hidden sm:inline">Xóa lịch sử</span>
        </button>
      </div>

      <div className="grid grid-cols-2 sm:grid-cols-4 md:grid-cols-8 gap-2.5">
        {history.map((item) => {
          const isPneumonia = item.predictionResult?.prediction === 'PNEUMONIA';
          const isCurrent =
            currentResult &&
            currentResult.prediction === item.predictionResult?.prediction &&
            currentResult.confidence === item.predictionResult?.confidence;

          return (
            <button
              key={item.id}
              type="button"
              onClick={() => onSelectHistoryItem(item)}
              className={`group relative rounded-xl overflow-hidden border-2 text-left transition-all cursor-pointer p-1.5 ${
                isCurrent
                  ? 'border-teal-500 bg-teal-50/40 dark:bg-teal-950/30'
                  : 'border-slate-200 dark:border-slate-700/80 hover:border-slate-300 dark:hover:border-slate-600 bg-slate-50/60 dark:bg-slate-800/40'
              }`}
            >
              <div className="aspect-square bg-slate-950 rounded-lg overflow-hidden relative mb-1.5">
                <img
                  src={item.previewUrl}
                  alt={item.fileName}
                  className="w-full h-full object-cover transition-transform duration-150 group-hover:scale-105"
                />
                <div className="absolute top-1 right-1">
                  {isPneumonia ? (
                    <span className="p-0.5 rounded bg-rose-600 text-white block">
                      <AlertTriangle className="h-2.5 w-2.5" />
                    </span>
                  ) : (
                    <span className="p-0.5 rounded bg-emerald-600 text-white block">
                      <ShieldCheck className="h-2.5 w-2.5" />
                    </span>
                  )}
                </div>
              </div>

              <div className="text-[11px] font-medium truncate text-slate-700 dark:text-slate-300">
                {item.fileName}
              </div>
              <div className="text-[10px] text-slate-400 font-mono">
                {item.timestamp}
              </div>
            </button>
          );
        })}
      </div>
    </div>
  );
}
