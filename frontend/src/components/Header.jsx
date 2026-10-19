import React from 'react';
import { Activity, ShieldCheck, AlertCircle, Cpu } from 'lucide-react';

export default function Header({ health, isCheckingHealth }) {
  const isHealthy = health?.status === 'healthy';

  return (
    <header className="border-b border-slate-200 bg-white/90 backdrop-blur-md sticky top-0 z-40">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex items-center justify-between h-20">
          {/* Logo & Tiêu đề */}
          <div className="flex items-center space-x-3.5">
            <div className="h-12 w-12 rounded-xl bg-gradient-to-br from-sky-500 to-blue-600 flex items-center justify-center shadow-lg shadow-sky-500/20 text-white ring-4 ring-sky-100">
              <Activity className="h-6 w-6 stroke-[2.2]" />
            </div>
            <div>
              <div className="flex items-center space-x-2">
                <h1 className="text-xl font-bold tracking-tight text-slate-900 sm:text-2xl">
                  Hỗ Trợ Sàng Lọc Viêm Phổi
                </h1>
                <span className="hidden sm:inline-flex items-center px-2 py-0.5 rounded text-xs font-semibold bg-sky-100 text-sky-800 border border-sky-200">
                  Nghiên Cứu
                </span>
              </div>
              <p className="text-xs text-slate-500 sm:text-sm font-medium">
                Hệ thống phân loại ảnh X-quang lồng ngực ứng dụng YOLOv8 & Tiền xử lý CLAHE
              </p>
            </div>
          </div>

          {/* Trạng thái Server & Thiết bị */}
          <div className="flex items-center space-x-3">
            {isCheckingHealth ? (
              <div className="flex items-center space-x-2 text-xs text-slate-400 bg-slate-100 px-3 py-1.5 rounded-full border border-slate-200">
                <span className="h-2 w-2 rounded-full bg-slate-400 animate-pulse"></span>
                <span>Đang kết nối...</span>
              </div>
            ) : isHealthy ? (
              <div className="flex items-center space-x-3">
                <div className="hidden md:flex items-center space-x-1.5 text-xs text-slate-600 bg-slate-50 px-2.5 py-1.5 rounded-lg border border-slate-200">
                  <Cpu className="h-3.5 w-3.5 text-slate-500" />
                  <span className="font-semibold uppercase text-slate-700">{health.device}</span>
                  <span className="text-slate-400">•</span>
                  <span>CLAHE: {health.use_clahe ? 'Bật' : 'Tắt'}</span>
                </div>
                <div className="flex items-center space-x-1.5 text-xs font-medium text-emerald-700 bg-emerald-50 px-3 py-1.5 rounded-full border border-emerald-200 shadow-sm">
                  <span className="relative flex h-2 w-2">
                    <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-emerald-400 opacity-75"></span>
                    <span className="relative inline-flex rounded-full h-2 w-2 bg-emerald-500"></span>
                  </span>
                  <span>Máy chủ sẵn sàng</span>
                </div>
              </div>
            ) : (
              <div className="flex items-center space-x-1.5 text-xs font-medium text-rose-700 bg-rose-50 px-3 py-1.5 rounded-full border border-rose-200">
                <AlertCircle className="h-3.5 w-3.5 text-rose-500" />
                <span>Mất kết nối backend</span>
              </div>
            )}
          </div>
        </div>
      </div>
    </header>
  );
}
