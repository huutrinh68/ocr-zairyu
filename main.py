# -*- coding: utf-8 -*-
from scipy import ndimage
import cv2
import unicodedata
import jaconv
import pytesseract

from utils import to_bin, denoise, \
    card_number, padding, filter_small_part, get_vertical_profile, showimg, remover, clearly

pytesseract.pytesseract.tesseract_cmd = "/usr/local/Cellar/tesseract/4.1.0/bin/tesseract"

hard_code = {
    'name': [0.07, 1, 1], 
    'date_of_birth': [0.11, 0.38, 2], 
    'gender': [0.38, 0.5, 2], 
    'nationality': [0.6, 0.81, 2], 
    'address': [0.1, 0.71, 4], 
    'period_of_stay': [0.22, 0.3, 7], 
    'type_permission': [0.14, 0.55, 9], 
    'permitted_date': [0.13, 0.36, 10],
    'expiration_date': [0.17, 0.52, 11]
    }

def another_try(path):
    info = {}
    #IMG_0872
    import matplotlib.image as mpimg
    clearly(path)
    showimg(to_bin(path), "binary")
    image = mpimg.imread(path)
    # img = cv2.imread("/Users/binhna/Downloads/cmnd2.jpg")
    showimg(image)
    img = remover(path, 85, 255)
    img = 255 - img
    showimg(img)
    
    
    pts = cv2.findNonZero(img)
    ret = cv2.minAreaRect(pts)
    (cx, cy), (w, h), ang = ret
    if w < h:
        w, h = h, w
        ang += 90
    print(cx, cy, w, h, ang)

    tmp = img.copy()
    tmp = cv2.cvtColor(tmp, cv2.COLOR_GRAY2BGR)
    cv2.rectangle(tmp, (int(cx), int(cy)), (int(cx+w), int(cy+h)), (0, 255, 0), 3)
    showimg(tmp)


    M = cv2.getRotationMatrix2D((cx, cy), ang, 1.0)
    rotated = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))
    showimg(rotated)

    # Resize to height = 1650 x width
    h, w = rotated.shape[:2]
    min_edge = 0 if h/w < 1 else 1
    factor = 1
    if rotated.shape[min_edge] < 1650:
        factor = 1650/rotated.shape[min_edge]
    rotated = cv2.resize(rotated, None, fx=factor, fy=factor,
                     interpolation=cv2.INTER_CUBIC)
    # print(img.shape)

    showimg(rotated)
    
    info['card_number'] = card_number(rotated)
    print("card number: {}".format(info['card_number']))

    ## (5) find and draw the upper and lower boundary of each lines
    # [:, rotated.shape[1]//5:int(rotated.shape[1]*2/5)]
    ## this tmp_img is for getting the lines of text
    tmp_img = rotated[:, rotated.shape[1]//5:int(rotated.shape[1]*3/5)]
    tmp_img = denoise(tmp_img)
    ## horizontal projection
    hist = cv2.reduce(tmp_img, 1, cv2.REDUCE_AVG).reshape(-1)
    
    th = 0
    H, W = tmp_img.shape[:2]
    uppers = [y for y in range(H-1) if hist[y] <= th and hist[y+1] > th]
    lowers = [y for y in range(H-1) if hist[y] > th and hist[y+1] <= th]
    print(uppers, lowers)
    
    ## make sure uppers and lowers have the same len
    len_min = min(len(uppers), len(lowers))
    uppers = uppers[:len_min]
    lowers = lowers[:len_min]

    img = 255 - rotated.copy()
    rotated = cv2.cvtColor(rotated, cv2.COLOR_GRAY2BGR)
    for y in uppers:
        cv2.line(rotated, (0, y), (rotated.shape[1], y), (255, 0, 0), 1)

    for y in lowers:
        cv2.line(rotated, (0, y), (rotated.shape[1], y), (0, 255, 0), 1)
    
    showimg(rotated)

    h, w = img.shape[:2]
    for key, value in hard_code.items():
        line = value[-1]
        if line >= len(uppers):
            break
        start = int(value[0]*w)
        end = int(value[1]*w)
        sub_img = img[uppers[line]:lowers[line], start:end]
        sub_img = padding(sub_img)
        lang = 'jpn_best' if line != 1 else 'eng'
        result = pytesseract.image_to_string(sub_img, lang=lang)
        result = unicodedata.normalize("NFKC", result)
        result = jaconv.normalize(result, 'NFKC')
        info[key] = result 
        # info['gender'] = '男' if 'M' in info['gender'] else '女'
        
    for key, value in info.items():
        # print(key, value)
        print(f"{key}: {value}")
    # for i in range(len(uppers)):
    #     sub_img = img[uppers[i]:lowers[i],:]
    #     sub_img = denoise(sub_img.copy())
    #     sub_img = 255 - sub_img
    #     padding = np.ones((50, sub_img.shape[1]))
    #     tmp_img = np.concatenate(
    #         (padding*255, sub_img, padding*255), axis=0)
    #     # showimg(tmp_img)
    #     lang = 'jpn_best' if i != 1 else 'eng'
    #     result = pytesseract.image_to_string(tmp_img, lang=lang)
    #     result = result.replace("ll", 'H').replace(
    #         "²", "2").replace("º", "o").replace("†", "1")
    #     print(f"{result}")
    #     if i == 3:
    #         print(f"ID Number: {pytesseract.image_to_string(255-sub_img, lang='vie')}")
    #     elif i == 4:
    #         print(f"Name: {pytesseract.image_to_string(255-sub_img, lang='vie')}")
    #     elif i == 5:
    #         print(
    #             f"Date of birth: {pytesseract.image_to_string(255-sub_img, lang='vie')}")

    # showimg(255-rotated)
    


    #showimg(255-img_)

# first_try()
if __name__ == '__main__':
    another_try("images/zairyuu5.jpg")