import cv2
import matplotlib.pyplot as plt
import numpy as np
import pytesseract


def to_bin(path):
    image = cv2.imread(path, 0)
    # image = cv2.adaptiveThreshold(image,20,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,67,2)
    _, image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    return image


def remover(path, low=0, high=255):
    image = cv2.imread(path, 0)
    print(image.mean())
    _, image = cv2.threshold(image, image.mean()/1.4, high, cv2.THRESH_BINARY)
    return image


def showimg(img, title=''):
    plt.imshow(img, cmap='gray')
    plt.title(title, fontproperties='')
    plt.show()


def get_vertical_profile(im):
    v_prof = np.sum((im), axis=1)
    smoothed_prof = smooth(v_prof, 9)
    plt.plot(smoothed_prof)
    plt.show()
    return smoothed_prof


def filter_small_part(img, _rotate):
    # showimg(img)
    x, y, width, height = cv2.boundingRect(img)
    # print(x,y,width,height)
    # print(img.shape)
    img = img[y:y+height, x:x+width]
    ret, labels, stats, centroids = cv2.connectedComponentsWithStats(
        img.astype(np.uint8), connectivity=8)
    new_img = np.zeros_like(img)

    # plt.imshow(img)
    # plt.show()

    for i in range(1, ret):
        cc_x = stats[i, cv2.CC_STAT_LEFT]
        cc_y = stats[i, cv2.CC_STAT_TOP]
        cc_width = stats[i, cv2.CC_STAT_WIDTH]
        cc_height = stats[i, cv2.CC_STAT_HEIGHT]

        if cc_width >= 0.2*width or cc_height >= 0.2*height:
            # if cc_width >= 0.1*width or cc_height >= 0.1*height:
            new_img[labels == i] = 1

        if cc_y <= height*0.2:
            print('SHAPE: ', width, height)
            print('CC: ', cc_width, cc_height)
            if cc_width >= 0.3*width and 0.05*height <= cc_height <= 0.2*height:
                _rotate = False
    return (new_img, _rotate)


def padding(sub_img):
    padding = np.ones((50, sub_img.shape[1]))
    tmp_img = np.concatenate((padding*255, sub_img, padding*255), axis=0)
    return tmp_img


def card_number(rotated):
    img_number = rotated[:rotated.shape[0]//3, int(rotated.shape[1]*3.5//5):]
    # showimg(img_number)

    # denoise some super big cc and super small cc
    ret, labels, stats, centroids = cv2.connectedComponentsWithStats(img_number, connectivity=8)
    for i in range(1, ret):
        if stats[i, cv2.CC_STAT_WIDTH] > 0.7*img_number.shape[1] or \
            stats[i, cv2.CC_STAT_WIDTH] < 15 and stats[i, cv2.CC_STAT_HEIGHT] < 15 or \
                stats[i, cv2.CC_STAT_TOP] < img_number.shape[0]*0.1:
            img_number[labels == i] = 0

    # horizontal projection 
    # showimg(img_number)
    hist = cv2.reduce(img_number, 1, cv2.REDUCE_AVG).reshape(-1)
    
    th = 0
    H, W = img_number.shape[:2]
    uppers = [y for y in range(H-1) if hist[y] <= th and hist[y+1] > th]
    lowers = [y for y in range(H-1) if hist[y] > th and hist[y+1] <= th]
    print(uppers, lowers)
    if uppers and lowers:
        uppers = uppers[0]
        lowers = lowers[0]
    else:
        print("Can't extract number ID")
        return ""
    img_number = 255 - img_number[uppers:lowers, :]
    # showimg(img_number)
    img_number = padding(img_number)
    showimg(img_number)
    result = pytesseract.image_to_string(img_number, lang='eng').split(' ')
    print(result)
    for r in result:
        if len(r) >= 12:
            return(r[-12:])
            #print(r[-12:])
    return ""


def denoise(rotated):
    # remove noise
    img_ = rotated.copy()
    img_debug = 255-img_
    # showimg(255-img_)
    ret, labels, stats, centroids = cv2.connectedComponentsWithStats(img_, connectivity=8)
    max_cc = max([stats[i, cv2.CC_STAT_HEIGHT] for i in range(1, ret)])
    for i in range(1, ret):
        x = stats[i, cv2.CC_STAT_LEFT]
        y = stats[i, cv2.CC_STAT_TOP]
        w = stats[i, cv2.CC_STAT_WIDTH]
        h = stats[i, cv2.CC_STAT_HEIGHT]
        #im2 = cv2.rectangle(img_debug, (x, y), (x+w, y+h), (0, 255, 0), 1)
        if stats[i, cv2.CC_STAT_WIDTH] < 15 and stats[i, cv2.CC_STAT_HEIGHT] < 15 \
            or stats[i, cv2.CC_STAT_WIDTH]/stats[i, cv2.CC_STAT_HEIGHT] > 10 or \
                stats[i, cv2.CC_STAT_TOP] > img_.shape[0]*0.9:
            img_[labels == i] = 0
    # showimg(im2)
    # showimg(255-img_)
    return img_


def clearly(path):
    img = cv2.imread(path, 1)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    gamma = 0.4 # r<1 will increase bright contrast ratio

    # Gamma adjustment
    pic1 = gray.max() * (gray/gray.max()) ** (1/gamma)
    # cv2.imwrite(os.path.join('output_image', '1_gamma_adjusted.png'), pic1)

    #### (2) Effect - Adaptive Binary Thresholding ####
    pic2 = cv2.adaptiveThreshold(gray,20,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,67,2)
    # cv2.imwrite(os.path.join('output_image','2_binary_thresholding.png'), pic2)

    pic3 = cv2.adaptiveThreshold(gray,20,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,33,2)
    # cv2.imwrite(os.path.join('output_image','3_binary_thresholding_gaussian.png'), pic3)

    #### (3) Effect - Laplacian Filter ####
    pic4 = cv2.Laplacian(gray,cv2.CV_16SC1, ksize=29)
    # cv2.imwrite(os.path.join('output_image','4_laplacian_filter.png'), pic4)

    titles = ['Original (Blurry Word)', 'Gamma Adjusted (r=0.4)','Adaptive Mean Thresholding (blocksize=67)', 
                'Adaptive Gaussian Thresholding', 'Laplacian (kernel size=29)']

    images = [img, pic1, pic2, pic3, pic4]

    for i in range(len(images)):
        plt.subplot(-(-len(images)//2),2,i+1),plt.imshow(images[i],'gray')
        plt.title(titles[i])
        plt.xticks([]),plt.yticks([])
    plt.show()
