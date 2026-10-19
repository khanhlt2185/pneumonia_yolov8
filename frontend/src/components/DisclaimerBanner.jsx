import React from 'react';
import { ShieldAlert } from 'lucide-react';

export default function DisclaimerBanner() {
  return (
    <aside aria-label="Lưu ý y khoa" className="bg-amber-50/90 border border-amber-200/90 rounded-xl p-4 shadow-sm">
      <div className="flex items-start space-x-3">
        <div className="p-1 rounded-lg bg-amber-100 text-amber-800 shrink-0 mt-0.5">
          <ShieldAlert className="h-5 w-5" />
        </div>
        <div className="text-xs sm:text-sm text-amber-950 leading-relaxed font-medium">
          <strong className="font-bold text-amber-900 block mb-0.5">Khuyến cáo y khoa quan trọng:</strong>
          Công cụ hỗ trợ sàng lọc phục vụ nghiên cứu, không thay thế chẩn đoán của bác sĩ. Mô hình được huấn luyện trên ảnh X-quang ngực bệnh nhi (1-5 tuổi) từ một cơ sở; chưa được kiểm chứng lâm sàng.
        </div>
      </div>
    </aside>
  );
}
