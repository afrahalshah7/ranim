# استخدم Python مناسب للصوت
FROM python:3.10

# مجلد العمل داخل الحاوية
WORKDIR /app

# نسخ الملفات
COPY . .

# تثبيت المتطلبات
RUN pip install --no-cache-dir fastapi uvicorn python-multipart \
    numpy librosa noisereduce scipy soundfile

# فتح البورت (مهم لـ Cloud Run)
EXPOSE 8080

# تشغيل FastAPI
CMD ["uvicorn", "main_last:app", "--host", "0.0.0.0", "--port", "8080"]