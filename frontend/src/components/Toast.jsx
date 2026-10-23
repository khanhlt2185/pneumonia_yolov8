import React, { useEffect } from 'react';
import { AlertCircle, CheckCircle, X } from 'lucide-react';

export default function Toast({ message, type = 'error', onClose }) {
  useEffect(() => {
    if (!message) return;
    const timer = setTimeout(() => {
      onClose();
    }, 5000);
    return () => clearTimeout(timer);
  }, [message, onClose]);

  if (!message) return null;

  const isError = type === 'error';

  return (
    <div className="fixed bottom-5 right-5 z-50 max-w-md animate-in fade-in slide-in-from-bottom-5 duration-200">
      <div
        className={`rounded-xl p-3.5 shadow-lg border flex items-start space-x-3 transition-colors ${
          isError
            ? 'bg-rose-50 dark:bg-slate-900 text-rose-900 dark:text-rose-200 border-rose-200 dark:border-rose-900/80 shadow-rose-900/10'
            : 'bg-emerald-50 dark:bg-slate-900 text-emerald-900 dark:text-emerald-200 border-emerald-200 dark:border-emerald-900/80 shadow-emerald-900/10'
        }`}
      >
        <div className="shrink-0 mt-0.5">
          {isError ? (
            <AlertCircle className="h-4 w-4 text-rose-600 dark:text-rose-400" />
          ) : (
            <CheckCircle className="h-4 w-4 text-emerald-600 dark:text-emerald-400" />
          )}
        </div>
        <div className="text-xs sm:text-sm font-medium leading-snug grow">
          {message}
        </div>
        <button
          type="button"
          onClick={onClose}
          className="text-slate-400 hover:text-slate-600 dark:hover:text-slate-200 p-0.5 rounded transition-colors shrink-0 cursor-pointer"
        >
          <X className="h-4 w-4" />
        </button>
      </div>
    </div>
  );
}
