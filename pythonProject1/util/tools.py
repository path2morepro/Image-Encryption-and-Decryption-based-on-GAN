import cv2
from matplotlib import pyplot as plt
from torchvision.utils import make_grid, save_image
import numpy as np
import matplotlib
matplotlib.use("Agg")

##加密函数：使用异或操作进行加密
# def encrypt(img, key):
#     encrypted_img = np.bitwise_xor(img, key)   #得到一个Numpy数组
#     # img = Image.fromarray(img)  #转换为PIL图像对象
#     return encrypted_img
#
# ###解密函数：使用异或操作进行解密
# def decrypt(encrypted_img, key):
#     img = np.bitwise_xor(encrypted_img, key)
#     decrypted_img = Image.fromarray(img)
#     return decrypted_img

##将一批图像数据进行处理并以指定的网络布局展示，将生成的图像网络保存到指定路径
def show_images(images, path, nrow=4, ncol=3):
    """
        展示图像网格。

        将给定的图像批处理显示为一个网格，每个图像占据网格中的一个单元格。
        参数:
        images: 图像批处理，包含多个图像的数据张量。
        path: 保存展示图像的文件路径。
        nrow: 网格中每行的图像数量，默认为4。
        ncol: 网格中每列的图像数量，默认为3。
        """
    # 计算网格中图像的数量
    num_images = nrow * ncol
    selected_images = images[:num_images]                 # 选择前num_images个图像进行展示

    # 重塑图像以进行可视化(假设图像的大小为 [C, H, W])
    selected_images = selected_images.cpu().data
    selected_images = selected_images[:, :, :, :]  # Optional, only if the batch size is not equal to nrow * ncol

    # 对图像数据进行归一化，使其值位于0到1之间
    selected_images = (selected_images+0.5)
    min_val = selected_images.min()
    max_val = selected_images.max()
    # 将像素值从[min_val, max_val] 缩收到[0, 1]
    selected_images = (selected_images - min_val) / (max_val - min_val)
    # Create a grid of images and display
    grid = make_grid(selected_images, nrow=nrow, ncol=ncol)
    npimg = grid.numpy()
    fig = plt.figure(figsize=(12, 12))  # 设置figsize以便显示的图像更大

    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.axis('off')  # Turn off axis labels
    plt.savefig(path)
    plt.close(fig)


# 将一系列处理过的图像保存为PNG格式的图片文件
def save_images(images, path, batch, isGT='rec'):
    for i, image in enumerate(images):
        image = image.cpu().data
        image = (image + 0.5)   #确保都在[0,1]之间，进行标准化处理
        min_val = image.min()
        max_val = image.max()
        # Rescale the pixel values from [min_val, max_val] to [0, 1]
        image = (image - min_val) / (max_val - min_val)   #归一化公式，把像素值重新缩放到[0,1]范围内
        image_path = os.path.join(path, f"{batch}_{i + 1}{isGT}.png")
        save_image(image, image_path)

        # fig = plt.figure(figsize=(6, 6))  # Set the figsize to make the displayed image larger
        # plt.imshow(np.transpose(npimg, (1, 2, 0)))
        # plt.axis('off')  # Turn off axis labels
        # plt.savefig(image_path)
        # plt.close(fig)


def makeDir(path):
    if not os.path.exists(path):
        os.makedirs(path)
### 将plt换成pillow
from PIL import Image
import os
def findImg(target1, target2, target3, path):
    """
       根据给定的三个目标图像，在指定路径下寻找与这些目标图像颜色特征相似的图像。

       参数:
       target1, target2, target3: 三个目标图像，用于比较颜色特征。
       path: 图像搜索的路径。

       返回:
       无返回值，但将找到的相似图像写入到指定的文件夹中。
       """
    target1 = cv2.cvtColor(np.array(target1), cv2.COLOR_RGB2BGR) # convert PIL image to OpenCV image
    target2 = cv2.cvtColor(np.array(target2), cv2.COLOR_RGB2BGR) # convert PIL image to OpenCV image
    target3 = cv2.cvtColor(np.array(target3), cv2.COLOR_RGB2BGR) # convert PIL image to OpenCV image

#计算目标图像的直方图
    target1_hist = cv2.calcHist([target1], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256]) # calculate the histogram of the target image
    target2_hist = cv2.calcHist([target2], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
    target3_hist = cv2.calcHist([target3], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])

#归一化直方图
    cv2.normalize(target1_hist, target1_hist, 0, 1, cv2.NORM_MINMAX) # normalize the histogram
    cv2.normalize(target2_hist, target2_hist, 0, 1, cv2.NORM_MINMAX) # normalize the histogram
    cv2.normalize(target3_hist, target3_hist, 0, 1, cv2.NORM_MINMAX) # normalize the histogram

#保存直方图的文件夹
    savepath = r'./result'
    makeDir(os.path.join(savepath, '493'))
    makeDir(os.path.join(savepath, '491'))
    makeDir(os.path.join(savepath, '487'))
    cnt = 0
    for root, dirs, files in os.walk(path):
        for file in files:
            cnt += 1
            print(cnt)

            ##将file读取为图片，计算直方图相似度
            img = cv2.imread(os.path.join(root, file))
            # img = cv2.resize(img, (256, 256)) # resize the image to the same size as the target image
            img_hist = cv2.calcHist([img], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256]) # calculate the histogram of the image
            cv2.normalize(img_hist, img_hist, 0, 1, cv2.NORM_MINMAX) # normalize the histogram
            similarity1 = cv2.compareHist(target1_hist, img_hist, cv2.HISTCMP_CORREL) # compare the histograms and get the similarity
            similarity2 = cv2.compareHist(target2_hist, img_hist, cv2.HISTCMP_CORREL) # compare the histograms and get the similarity
            similarity3 = cv2.compareHist(target3_hist, img_hist, cv2.HISTCMP_CORREL) # compare the histograms and get the similarity
            if similarity1 > 0.9:

                cv2.imwrite(os.path.join(savepath, '493', file), img)
            elif similarity2 > 0.9:
                cv2.imwrite(os.path.join(savepath, '491', file), img)
            elif similarity3 > 0.9:
                cv2.imwrite(os.path.join(savepath, '487', file), img)
