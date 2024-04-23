import os
d = os.listdir('/Users/guohongcheng/Downloads/mediapipe-master/bounding_box_imgs')
with open('mediapipe/models/knift_plu_labelmap.txt', 'w') as fp:
    for i in d:
        if i.endswith('.png'):
            fp.write(i[:-4] + '\n')

os.system("echo 'mediapipe/models/knift_plu_labelmap.txt 256'")
os.system('shasum -a 256 mediapipe/models/knift_plu_labelmap.txt')
