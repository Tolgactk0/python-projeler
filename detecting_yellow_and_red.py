import cv2
import numpy as np

# Resmi yükleme
img=cv2.imread('k.png')# resmi yükleme

# Resmin doğru yüklenip yüklenmediğini kontrol etme
if img is None:
    print("Resim yüklenemedi. Dosya yolunu kontrol edin.")
else:
    # Resmi HSV renk uzayına çevirme
    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Kırmızı renk aralığı
    low_red1 = np.array([0, 50, 70])
    high_red1 = np.array([10, 255, 255])
    low_red2 = np.array([170, 50, 70])
    high_red2 = np.array([180, 255, 255])

    # Kırmızı renk için maske oluşturma
    mask_red1 = cv2.inRange(hsv_img, low_red1, high_red1)
    mask_red2 = cv2.inRange(hsv_img, low_red2, high_red2)
    mask_red = cv2.add(mask_red1, mask_red2)
    mask_red = cv2.erode(mask_red, None, iterations=2)
    mask_red = cv2.dilate(mask_red, None, iterations=2)
    mask_red = cv2.GaussianBlur(mask_red, (3, 3), 0)

    # Kırmızı renk konturlarını bulma
    cnts_red = cv2.findContours(mask_red.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]

    # Sarı renk aralığı
    low_yellow = np.array([20, 100, 100])
    high_yellow = np.array([30, 255, 255])

    # Sarı renk için maske oluşturma
    mask_yellow = cv2.inRange(hsv_img, low_yellow, high_yellow)
    mask_yellow = cv2.erode(mask_yellow, None, iterations=2)
    mask_yellow = cv2.dilate(mask_yellow, None, iterations=2)
    mask_yellow = cv2.GaussianBlur(mask_yellow, (3, 3), 0)

    # Sarı renk konturlarını bulma
    cnts_yellow = cv2.findContours(mask_yellow.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]

    # Kırmızı konturları çizme
    for cnt in cnts_red:
        (x, y, w, h) = cv2.boundingRect(cnt)
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 4)  # Kırmızı renkte dikdörtgen

    # Sarı konturları çizme
    for cnt in cnts_yellow:
        (x, y, w, h) = cv2.boundingRect(cnt)
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 255), 4)  # Sarı renkte dikdörtgen

    # Görüntüyü gösterme
    cv2.imshow("kamera", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
