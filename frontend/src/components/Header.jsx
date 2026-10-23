import React from 'react';
import { Activity, AlertCircle, Sun, Moon } from 'lucide-react';

export default function Header({ health, isCheckingHealth, isDark, onToggleTheme }) {
  const isHealthy = health?.status === 'healthy';

  return (
    <header className="border-b border-slate-200 dark:border-slate-800 bg-white/95 dark:bg-slate-900/95 sticky top-0 z-30 transition-colors">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex items-center justify-between h-16 sm:h-18">
          {/* Logo & Tiêu đề */}
          <div className="flex items-center space-x-3">
            <div className="h-10 w-10 rounded-xl bg-teal-600 dark:bg-teal-500 flex items-center justify-center text-white shadow-sm">
              <Activity className="h-5 w-5 stroke-[2.2]" />
            </div>
            <div>
              <h1 className="text-lg sm:text-xl font-bold tracking-tight text-slate-900 dark:text-slate-100">
                Hỗ Trợ Sàng Lọc Viêm Phổi
              </h1>
              <p className="text-xs text-slate-500 dark:text-slate-400">
                Phân loại hình ảnh X-quang lồng ngực
              </p>
            </div>
          </div>

          {/* Trạng thái máy chủ & Chuyển đổi Theme */}
          <div className="flex items-center space-x-3 sm:space-x-4">
            {/* Trạng thái server đơn giản, không lộ thông số kỹ thuật */}
            {isCheckingHealth ? (
              <div className="flex items-center space-x-1.5 text-xs text-slate-500 dark:text-slate-400 bg-slate-100 dark:bg-slate-800 px-3 py-1.5 rounded-full border border-slate-200 dark:border-slate-700">
                <span className="h-2 w-2 rounded-full bg-slate-400 animate-pulse" />
                <span>Đang kết nối...</span>
              </div>
            ) : isHealthy ? (
              <div className="flex items-center space-x-1.5 text-xs font-medium text-emerald-700 dark:text-emerald-400 bg-emerald-50 dark:bg-emerald-950/60 px-3 py-1.5 rounded-full border border-emerald-200 dark:border-emerald-800/80">
                <span className="relative flex h-2 w-2">
                  <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-emerald-400 opacity-75" />
                  <span className="relative inline-flex rounded-full h-2 w-2 bg-emerald-500" />
                </span>
                <span>Máy chủ sẵn sàng</span>
              </div>
            ) : (
              <div className="flex items-center space-x-1.5 text-xs font-medium text-rose-700 dark:text-rose-400 bg-rose-50 dark:bg-rose-950/60 px-3 py-1.5 rounded-full border border-rose-200 dark:border-rose-800/80">
                <AlertCircle className="h-3.5 w-3.5 text-rose-500" />
                <span>Mất kết nối máy chủ</span>
              </div>
            )}

            {/* Nút chuyển đổi Light / Dark mode */}
            <button
              id="btn-toggle-theme"
              type="button"
              onClick={onToggleTheme}
              aria-label={isDark ? 'Chuyển sang giao diện sáng' : 'Chuyển sang giao diện tối'}
              className="p-2 rounded-xl text-slate-600 dark:text-slate-300 bg-slate-100 dark:bg-slate-800 hover:bg-slate-200 dark:hover:bg-slate-700 transition-colors cursor-pointer border border-slate-200 dark:border-slate-700 focus:outline-none focus:ring-2 focus:ring-teal-500"
            >
              {isDark ? (
                <Sun className="h-4 w-4 text-amber-400" />
              ) : (
                <Moon className="h-4 w-4 text-slate-600" />
              )}
            </button>
          </div>
        </div>
      </div>
    </header>
  );
}
