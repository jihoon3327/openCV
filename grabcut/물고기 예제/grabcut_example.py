import sys
import numpy as np
import cv2

src = cv2.imread('opencv/nemo.jpg')

if src is None:
    print('image load failed')
    sys.exit()

#사각형 선택
rc = cv2.selectROI(src)

#선택되지 않은 부분 블랙 처리
mask = np.zeros(src.shape[:2], np.uint8)

cv2.grabCut(src, mask, rc, None, None, 5, cv2.GC_INIT_WITH_RECT)

# 0 : cv2.GC_BGD, 2: cv2.GC_PR_RGD
mask_fg = np.where( (mask == 0) | (mask == 2), 0, 1).astype('uint8')
mask_bg = np.where( (mask == 1) | (mask == 3), 0, 1).astype('uint8')
cv2.imshow('mask_fg', mask_fg*255)
cv2.imshow('mask_bg', mask_bg*255)


dst_fg = src * mask_fg[:, :, np.newaxis]
dst_bg = src * mask_bg[:, :, np.newaxis]

cv2.imshow('dst_fg', dst_fg)
cv2.imshow('dst_fg', dst_fg)

cv2.waitKey()
cv2.destroyAllWindows()