import cv2 as cv 
import numpy as np 

images = []
exposure_times = np.array([], dtype=np.float32)


# merge

merge_debevec = cv.createMergeDebevec()
hdr_debevec = merge_debevec.process(images, times=exposure_times.copy())

# tonemap

tonemap = cv.createTonemap(gamma= 2.2)
res_debevec = tonemap.process(hdr_debevec.copy())


res_debevec_8bit = np.clip(res_debevec*255, 0, 255).astype('uint8')

cv.imwrite("", res_debevec_8bit)
