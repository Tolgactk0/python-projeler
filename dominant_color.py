import cv2
from sklearn.cluster import KMeans
import numpy as np

def getdcolor(img, n):
    img = img.reshape((img.shape[0] * img.shape[1], 3))
    kmeans = KMeans(n_clusters=n)
    kmeans.fit(img)
    colors = kmeans.cluster_centers_
    return colors.astype(int)

img = cv2.imread('renk.jpg')
cluster = 6  # Toplam 6 renk bulunacak şekilde güncellendi

colors = getdcolor(img, cluster)
print(colors)

# Colorbar oluştururken 6 renge göre güncellendi
yeni = np.zeros((500, 600, 3), dtype=np.uint8)
yeni[:, :100] = colors[0]
yeni[:, 100:200] = colors[1]
yeni[:, 200:300] = colors[2]
yeni[:, 300:400] = colors[3]
yeni[:, 400:500] = colors[4]
yeni[:, 500:] = colors[5]  # Yeni 6. rengi burada ekledik

cv2.imshow("kamera", img)
cv2.imshow("Colorbar", yeni)
cv2.waitKey()
cv2.destroyAllWindows()
