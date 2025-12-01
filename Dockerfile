# استفاده از ایمیج اوبونتو 24.04 (با پایتون 3.12 و CUDA 12.6)
FROM nvidia/cuda:12.6.1-cudnn-devel-ubuntu24.04

# تنظیمات محیطی
ENV DEBIAN_FRONTEND=noninteractive
# اجازه نصب پکیج روی پایتون سیستمی
ENV PIP_BREAK_SYSTEM_PACKAGES=1

# نصب پکیج‌های سیستمی
RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-dev \
    build-essential \
    git \
    wget \
    && rm -rf /var/lib/apt/lists/*

# لینک کردن python به python3
RUN ln -s /usr/bin/python3 /usr/bin/python

WORKDIR /workspace

# کپی کردن فایل requirements
COPY requirements.txt .

# --- تغییر مهم: حذف دستور upgrade pip که باعث ارور می‌شد ---
# نسخه پیش‌فرض pip روی اوبونتو 24 کاملاً کافی است.
RUN python -m pip install --no-cache-dir -r requirements.txt

# نصب دستی پکیج‌های حیاتی اگر در requirements نبودند
RUN pip install --no-cache-dir sionna tensorflow[and-cuda] jupyterlab matplotlib pandas scipy

COPY . .

CMD ["/bin/bash"]