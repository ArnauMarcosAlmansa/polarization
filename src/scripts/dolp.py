import cv2
import matplotlib.pyplot as plt
import numpy as np

I0 = cv2.imread("../../data/video4-frame1/00001_000.png")[:, :, 0].astype(np.float32) / 255
I45 = cv2.imread("../../data/video4-frame1/00001_045.png")[:, :, 0].astype(np.float32) / 255
I90 = cv2.imread("../../data/video4-frame1/00001_090.png")[:, :, 0].astype(np.float32) / 255
I135 = cv2.imread("../../data/video4-frame1/00001_135.png")[:, :, 0].astype(np.float32) / 255

S0 = (I0 + I45 + I90 + I135) / 2
S1 = (I0 - I90)
S2 = (I45 - I135)

DoLP = np.sqrt(S1 ** 2 + S2 ** 2) / S0

plt.imshow(DoLP, cmap='gray')
plt.show()

print()