import React, { useEffect } from 'react';
import { AlertCircle, CheckCircle, Info, X } from 'lucide-react';

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
        className={`rounded-xl p-4 shadow-xl border flex items-start space-x-3 ${
          isError
            ? 'bg-rose-50 text-rose-900 border-rose-200 shadow-rose-500/10'
            : 'bg-emerald-50 text-emerald-900 border-emerald-200 shadow-emerald-500/10'
        }`}
      >
        <div className="shrink-0 mt-0.5">
          {isError ? (
            <AlertCircle className="h-5 w-5 text-rose-600" />
          ) : (
            <CheckCircle className="h-5 w-5 text-emerald-600" />
          )}
        </div>
        <div className="text-xs sm:text-sm font-medium leading-snug grow">
          {message}
        </div>
        <button
          onClick={onClose}
          className="text-slate-400 hover:text-slate-600 p-0.5 rounded transition-colors shrink-0"
        >
          <X className="h-4 w-4" />
        </button>
      </div>
    </div>
  );
}
