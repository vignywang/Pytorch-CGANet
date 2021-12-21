import random
from skimage import morphology
from data import *
def argument(img,mask):
    img, mask = randomHorizontalFlip(img, mask)
    img, mask = randomVerticleFlip(img, mask)
    img, mask = randomRotate90(img, mask)
    return img,mask
def extract_each_layer(image, threshold):
    """
    This image processing funtion is designed for the OCT image post processing.
    It can remove the small regions and find the OCT layer boundary under the specified threshold.
    :param image:
    :param threshold:
    :return:
    """
    # convert the output to the binary image
    ret, binary = cv2.threshold(image, threshold, 1, cv2.THRESH_BINARY)

    bool_binary = np.array(binary, bool)

    # remove the small object
    remove_binary = morphology.remove_small_objects(bool_binary, min_size=25000,
                                                                connectivity=2,
                                                                in_place=False)
    c = np.multiply(bool_binary, remove_binary)
    final_binary = np.zeros(shape=np.shape(binary))
    final_binary[c == True] = 1
    binary_image = cv2.filter2D(final_binary, -1, np.array([[-1], [1]]))
    layer_one = np.zeros(shape=[1, np.shape(binary_image)[1]])
    for i in range(np.shape(binary_image)[1]):
        location_point = np.where(binary_image[:, i] > 0)[0]
        # print(location_point)

        if len(location_point) == 1:
            layer_one[0, i] = location_point
        elif len(location_point) == 0:
            layer_one[0, i] = layer_one[0, i-1]

        else:
            layer_one[0, i] = location_point[0]

    return layer_one
def getpatch4(pre,ground_truth,mask,img):
    C = abs(ground_truth-pre)
    y = np.array([0, 1, 2, 3, 4])
    x = np.array([0, 1, 2, 3, 4])
    y[0] = 0
    y[1] = 544 / 4
    y[2] = y[1] * 2
    y[3] = y[1] * 3
    y[4] = y[1] * 4
    x[0] = 0
    x[1] = 544 / 4
    x[2] = x[1] * 2
    x[3] = x[1] * 3
    x[4] = x[1] * 4
    D = []
    for i in range(len(x) - 1):
        for j in range(len(y) - 1):
            D.append([C[x[i]:x[i + 1], y[j]:y[j + 1]].mean()+i,i,j])
    D.sort(reverse=True)
    arg_mask = np.zeros((544, 544), np.uint8)
    arg_image = np.zeros((544, 544, 3), np.uint8)
    for i in range(len(x) - 1):
        for j in range(len(y) - 1):
            id = random.randint(0, 3)
            D1 = D[id]
            a = D1[1]
            b = D1[2]
            patch_img,patch_mask=argument(img[x[a]:x[a + 1], y[b]:y[b + 1]],mask[x[a]:x[a + 1], y[b]:y[b + 1]])
            arg_image[x[i]:x[i + 1], y[j]:y[j + 1]] = patch_img
            arg_mask[x[i]:x[i + 1], y[j]:y[j + 1]] = patch_mask
    mask1 = Image.fromarray(arg_mask.astype('uint8'))
    image1 = Image.fromarray(arg_image.astype('uint8'))
    return image1,mask1

