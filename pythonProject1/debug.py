import os
from tqdm import tqdm
import numpy as np
import torch
from torch import optim, nn
from dataLoader.GetDataLoader import GetDataLoader
from model.model import UNET, Discriminator, Noise
import matplotlib.pyplot as plt
from model.losses import PerceptualLoss, KeyLoss, CorrelationLoss
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim


os.environ['CUDA_LAUNCH_BLOCKING'] = '1'


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 获取训练参数
train_data_path = "D:\\dataset\\subset"
batch_size = 4
num_epochs = 50
save_epoch = 10
print_epoch = 1
valid_epoch = 10
image_size = 256

# 设置随机种子，确保每次的key都一致
np.random.seed(443)

noise = Noise(input_dim=1).to(device)
unet = UNET().to(device) # 
generator = UNET().to(device)
discriminator = Discriminator().to(device)

unet.train()
generator.train()
noise.train()
discriminator.train()

unet_optimizer = optim.Adam(unet.parameters(), lr=0.0002, betas=(0.5, 0.999))
noise_optimizer = optim.Adam(noise.parameters(), lr=0.0002, betas=(0.5, 0.999))
generator_optimizer = optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
discriminator_optimizer = optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))


# loss
L1_loss = nn.L1Loss().to(device)
MSE_loss = nn.MSELoss().to(device)
perceptual_loss = PerceptualLoss().to(device)
key_loss_func = KeyLoss().to(device)
adversarial_loss = nn.BCELoss().to(device)                     # 判别器损失函数
pc_loss = CorrelationLoss().to(device)


train_hist = {}
train_hist['E_losses'] = []
train_hist['D_losses_MSE'] = []
train_hist['D_losses_L1'] = []
train_hist['total_losses'] = []
train_hist['d_loss_real'] = []
train_hist['d_loss_fake'] = []
train_hist['adv_loss'] = []
train_hist['encrypt_loss'] = []
train_hist['decrypt_loss_l1'] = []


def showImage(image, name, batch_size):
    image = image.cpu().squeeze().detach().numpy()  # 删除维度为1的维度
    if batch_size > 1:
        # 显示图像
        for i in range(image.shape[0]):
            name = 'subset\\' + name +'_' + str(i) +'.png'
            plt.imshow(image[i], cmap='gray')  # 使用灰度颜色图显示
            plt.axis('off')  # 隐藏坐标轴
 
            plt.savefig(name,bbox_inches='tight', pad_inches=0)

            
    else:
        name = 'subset\\' + name +'.png'
        plt.imshow(image, cmap='gray')
        plt.axis('off')  # 隐藏坐标轴
        plt.savefig(name, bbox_inches='tight', pad_inches=0)





# # 绘制并保存损失值曲线图
def plot_loss_curves(train_hist, model_save_path):
    for key, values in train_hist.items():
        plt.figure()
        plt.plot(values, label=key)
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.title(f'{key} Curve')
        plt.savefig(os.path.join(model_save_path, f'{key}_curve.png'))
        plt.close()



lambda_mse = 0.1  # 初始化MSE损失的权重，这些值需要根据模型表现进行调整
lambda_l1 = 0.4  # 初始化L1损失的权重
lambda_key = 0.1  # 初始化KeyLoss的权重
lambda_perceptual = 0.4  # 初始化感知损失的权重
        

def train(train_loader):

    for epoch in tqdm(range(num_epochs), desc="Epochs"):
        train_data_size = len(train_loader)
        # train_data = tqdm(train_loader, total=train_data_size, initial=1, leave=False, dynamic_ncols=True,
        #               desc=f"Epoch {epoch + 1}/{num_epochs}")

        epoch_encrypt_loss = 0
        epoch_discriminator_loss = 0
        epoch_total_loss = 0
        epoch_adv_loss = 0
        epoch_perceptual_loss = 0

        count = 0
        executed = True
        for x, key in tqdm(train_loader, desc="Batches", leave=False):
            count += 1
            x = x.to(device)
            key = key.to(device)

           

            # encoder的输出是两个，前面一个是起到一个降噪的作用，后面一个图片会先降噪然后加噪
            # 如果这里在编码器要求加上key，同时解决这个相似度分布问题，可以在encoder后面加一个unet来对图片进行编码加密

            encrypt_input, _  = noise(torch.cat([x, key], dim=1))
            encrypted_image = unet(encrypt_input)
            decrypted_input, _ = noise(torch.cat([encrypted_image, key], dim=1)) #只将channel不加噪
            decrypted_image = generator(decrypted_input)

        

            real_outputs = discriminator(x)                    
            fake_outputs = discriminator(decrypted_image.detach()) # 解码后的图片和原图放入判别器


            real_labels = torch.ones((real_outputs.size(0), 1), requires_grad=False).to(device)
            fake_labels = torch.zeros((fake_outputs.size(0), 1), requires_grad=False).to(device)
            
            d_loss_real = adversarial_loss(real_outputs, real_labels)
            d_loss_fake = adversarial_loss(fake_outputs, fake_labels)

            
            discriminator_optimizer.zero_grad()
            d_loss = (d_loss_real + d_loss_fake) / 2
            d_loss.backward()
            discriminator_optimizer.step()


            #问题：
            # discriminator的损失太小了
            # adv_loss又降不下来
            # generator太弱了，discriminator一下就分别出来了，所以损失很小
            # 加密网络训练
            adv_loss = adversarial_loss(discriminator(decrypted_image.detach()), real_labels) # generator生成的图像要尽可能和原图靠拢
            encrypt_loss = pc_loss(encrypted_image) # loss of encoder， 加密图像-->随机生成的密匙
            

            decrypt_loss_l1 = L1_loss(decrypted_image, x) # loss of decoder 解密图像尽可能向原图靠拢
            decrypt_loss_pc = pc_loss(decrypted_image)

            perceptual_loss_value = perceptual_loss(decrypted_image, x) # loss of decoder based on vgg

            # key_loss = key_loss_func(sharp, decrypted_image, wrong_key) # loss of key

            total_loss = decrypt_loss_l1 + decrypt_loss_pc + adv_loss + perceptual_loss_value

            # lambda_mse = 0.1  # 初始化MSE损失的权重，这些值需要根据模型表现进行调整
            # lambda_l1 = 0.4  # 初始化L1损失的权重
            # lambda_key = 0.1  # 初始化KeyLoss的权重
            # lambda_perceptual = 0.4  # 初始化感知损失的权重

            # 反向传播和优化
            noise_optimizer.zero_grad()
            unet.zero_grad()
            encrypt_loss.backward(retain_graph=True)
            
            
            generator_optimizer.zero_grad()
            total_loss.backward(retain_graph=False)
            generator_optimizer.step()
            unet_optimizer.step()
            noise_optimizer.step()

            epoch_encrypt_loss += encrypt_loss.item()
            epoch_discriminator_loss += d_loss.item()
            epoch_total_loss += total_loss.item()
            epoch_adv_loss += adv_loss.item()
            epoch_perceptual_loss += perceptual_loss_value.item()

            torch.cuda.empty_cache()
            # tqdm.write(f"Epoch [{epoch+1}/{num_epochs}], Batch Losses: adv_loss={adv_loss.item():.4f}, "
            #         f"encrypt_loss={encrypt_loss.item():.4f}, decrypt_loss_l1={decrypt_loss_l1.item():.4f}, "
            #         f"perceptual_loss={perceptual_loss_value.item():.4f},"
            #         f"discriminator_loss={d_loss.item():.4f},"
            #         f"total_loss={total_loss.item():.4f}")

            
            originals = np.moveaxis(x.cpu().numpy(), 1, -1)  # 源代码是1，-1
            reconstructions = np.moveaxis(decrypted_image.detach().cpu().numpy(), 1, -1)

            # 动态调整 win_size
            min_size = min(originals.shape[1:3])  # 获取图像的最小尺寸
            win_size = min(7, min_size)  # 确保 win_size 不超过图像的最小尺寸且至少为7

            


            
            if epoch % 5 == 0:
                en_img_name = 'encrypt_epoch' + str(epoch) + '_batch'
                de_img_name = 'decrypt_epoch' + str(epoch) + '_batch'
                x_image_name = 'epoch' + str(epoch) + '_batch'
                showImage(encrypted_image[0], en_img_name, 1)
                showImage(decrypted_image[0], de_img_name, 1)
                showImage(x[0], x_image_name, 1)
                
                # 计算 PSNR 和 SSIM
                epoch_psnr = 0.0
                epoch_ssim = 0.0
                for i in range(originals.shape[0]):
                    # For color images, set multichannel=True
                    epoch_psnr += psnr(originals[i], reconstructions[i], data_range=1.0)
                    # valid_ssim += ssim(valid_originals[i], valid_reconstructions[i], data_range=1.0, multichannel=True)  # 源代码
                    epoch_ssim += ssim(originals[i], reconstructions[i], data_range=1.0, win_size=win_size,
                                    channel_axis=-1)
                    if executed:
                        print("epoch {}, PSNR: {:.4f}, SSIM: {:.4f}".format(epoch + 1, epoch_psnr / originals.shape[0], epoch_ssim / originals.shape[0]))
                        executed = False


            
      
        epoch_encrypt_loss /= train_data_size
        epoch_discriminator_loss /= train_data_size
        epoch_total_loss /= train_data_size
        epoch_adv_loss /= train_data_size
        epoch_perceptual_loss /= train_data_size
        tqdm.write(f"Epoch [{epoch+1}/{num_epochs}], Average Losses: adv_loss={epoch_adv_loss:.4f}, "
                    f"encrypt_loss={epoch_encrypt_loss:.4f}, "
                    f"perceptual_loss={epoch_perceptual_loss:.4f},"
                    f"discriminator_loss={epoch_discriminator_loss:.8f},"
                    f"total_loss={epoch_total_loss:.4f}")
            

    

if __name__ == '__main__':

    training_loader = GetDataLoader(train_data_path, image_size, batch_size=batch_size)

    train(training_loader)
    torch.save(generator, 'generator.pth')
    torch.save(discriminator, 'discriminator.pth')



