from cypath.data.wsi import KFB
from PIL import Image
import time
start = time.time()
path = '/raid10/_datasets/Breast/QiLu/Tumor/00026.18.kfb'

# test a single image
max_mag = 40
curr_mag = 10
save_mag = 2.5
psize = 512
scale = curr_mag / save_mag
scale = int(scale)
ssize = psize // scale

wsi = KFB(path, max_mag, curr_mag)
wsi.setIterator(psize)

w,h = wsi.getSize()

x = wsi.x
y = wsi.y

for img in wsi:
    img = Image.fromarray(img)
    img.save(f'/tmp/a/a_{x}_{y}.jpg')
    x = wsi.x
    y = wsi.y
end = time.time()

print('time', (end-start)/60)
