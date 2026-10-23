import React from 'react';
import { ShieldAlert } from 'lucide-react';

export default function DisclaimerBanner() {
  return (
    <aside
      aria-label="Lưu ý y khoa"
      className="bg-amber-50 dark:bg-amber-950/40 border border-amber-200 dark:border-amber-900/60 rounded-xl p-3.5 shadow-xs transition-colors"
    >
      <div className="flex items-center space-x-3">
        <div className="p-1.5 rounded-lg bg-amber-100 dark:bg-amber-900/60 text-amber-800 dark:text-amber-300 shrink-0">
          <ShieldAlert className="h-4 w-4" />
        </div>
        <p className="text-xs sm:text-sm text-amber-900 dark:text-amber-200 font-medium leading-relaxed">
          <strong className="font-semibold text-amber-950 dark:text-amber-100">Khuyến cáo: </strong>
          Công cụ hỗ trợ sàng lọc phục vụ nghiên cứu, không thay thế chẩn đoán của bác sĩ.
        </p>
      </div>
    </aside>
  );
}
