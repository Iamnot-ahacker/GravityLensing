import cv2
import numpy as np

# Resimleri yükle (PNG formatı)
image1 = cv2.imread('.\GravityLensing\image1.png')  # Dosya yolunu kontrol edin
image2 = cv2.imread('.\GravityLensing\image2.png')  # Dosya yolunu kontrol edin

# Resimler doğru şekilde yüklenip yüklenmediğini kontrol et
if image1 is None:
    print("image1.png dosyası yüklenemedi!")
    exit()

if image2 is None:
    print("image2.png dosyası yüklenemedi!")
    exit()

# Resimleri aynı boyutta yap (gerekirse)
image1 = cv2.resize(image1, (image2.shape[1], image2.shape[0]))

# Farklılıkları hesapla
difference = cv2.absdiff(image1, image2)

# Farkları gri tonlamada görselleştir
gray_difference = cv2.cvtColor(difference, cv2.COLOR_BGR2GRAY)

# Farklılıkları eşik değerle
_, thresh = cv2.threshold(gray_difference, 30, 255, cv2.THRESH_BINARY)

# Farklılıkların sayısını belirle
non_zero_count = cv2.countNonZero(thresh)

# Eğer fark yoksa, resimler aynı demektir
if non_zero_count == 0:
    print("Resimler aynı.")
else:
    print(f"Resimler farklı. Farklılık sayısı: {non_zero_count}")
