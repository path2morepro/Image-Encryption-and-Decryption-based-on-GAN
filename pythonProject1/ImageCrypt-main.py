import configparser
import os
from tqdm import tqdm
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
import torch
from torch import optim, nn
from util.tools import makeDir, show_images, save_images
from dataLoader.GetDataLoader import GetDataLoader
from model.ImageCrypt import ImageCrypt
import torch.nn.functional as F
import torchvision.models as models
from torchvision.models import vgg16
from torchvision.models import vgg16, VGG16_Weights
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
from model.losses import PerceptualLoss, KeyLoss
# from model.losses import HistogramLoss，PerceptualLoss, KeyLoss, HistogramEntropyLoss, DiffusionLoss, CorrelationLoss, SimilarityLoss, ErrorLoss
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

from torch.utils.tensorboard import SummaryWriter

# 初始化SummaryWriter
writer = SummaryWriter('runs/experiment_name')

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
cfg_path = r'./config/ImageCrypt.cfg'
config = configparser.ConfigParser()
config.read(cfg_path)

# 获取训练参数
training_params = config['Training']
checkpoints_name = training_params['checkpoints_name']
train_data_path = training_params['train_data_path']
batch_size = int(training_params['batch_size'])
num_epochs = int(training_params['num_epochs'])
save_epoch = int(training_params['save_epoch'])
print_epoch = int(training_params['print_epoch'])
valid_epoch = int(training_params['valid_epoch'])
image_size = int(training_params['image_size'])

# 设置随机种子，确保每次的key都一致
np.random.seed(443)
model = ImageCrypt(input_dim=1, num_filter=64, output_dim=1).to(device)
model.train()
optimizer = optim.Adam(model.parameters(), lr=0.0002, betas=(0.5, 0.999))

# 创建 SummaryWriter 对象
# writer = SummaryWriter(log_dir=f'./logs/{checkpoints_name}')

num_steps = 10
# loss
L1_loss = nn.L1Loss().to(device)
MSE_loss = nn.MSELoss().to(device)
perceptual_loss = PerceptualLoss().to(device)
key_loss_func = KeyLoss().to(device)
adversarial_loss = nn.BCELoss().to(device)                          # 判别器损失函数
# histogram_entropy_loss = HistogramLoss(num_steps=num_steps).to(device)
# histogram_entropy_loss = HistogramLoss().to(device)          # 直方图和熵损失

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

lambda_mse = 0.3  # 初始化MSE损失的权重，这些值需要根据模型表现进行调整
lambda_l1 = 0.2  # 初始化L1损失的权重
lambda_key = 0.1  # 初始化KeyLoss的权重
lambda_perceptual = 0.2  # 初始化感知损失的权重
lambda_hist_ent = 1  # 初始化直方图损失的权重

lambda_diffusion = 0.1  # 初始化扩散损失的权重
lambda_correlation = 1  # 初始化像素相关性损失的权重
lambda_similarity = 0.2  # 初始化相似性损失的权重
lambda_error = 0.1  # 初始化错误损失的权重

# 判别器优化器
discriminator_optimizer = optim.Adam(model.discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))


# 绘制并保存损失值曲线图
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


# # 训练
def train_epoch(epoch, training_loader, validation_loader, save_path, encrypted_quantized, decrypted_quantized):

    train_data_size = len(training_loader)
    train_data = tqdm(training_loader, total=train_data_size, initial=1, leave=False,
                      desc=f"Epoch {epoch + 1}/{num_epochs}")

    for sharp, key in iter(train_data):
        sharp = sharp.to(device)
        key = key.to(device)
        # 生成错误密钥
        wrong_key = torch.randint(0, 256, key.shape, dtype=torch.float).to(device) / 255.0  # 和正确密钥生成方式一样

        optimizer.zero_grad()
        discriminator_optimizer.zero_grad()

        # 前向传播   # 生成N图像（加密网络生成的图像）
        encrypted_image, decrypted_image = model(sharp, key)
        # encrypted_image, decrypted_image, _, _ = model(sharp, key, encrypted_quantized, decrypted_quantized)
        # _, decrypted_wrong = model(sharp, wrong_key)

        # 判别器训练
        real_outputs = model.discriminator(key)                    # 原始密钥作为‘真’图像输入判别器
        fake_outputs = model.discriminator(encrypted_image.detach())

        real_labels = torch.ones_like(real_outputs).to(device)
        fake_labels = torch.zeros_like(fake_outputs).to(device)

        d_loss_real = adversarial_loss(real_outputs, real_labels)
        d_loss_fake = adversarial_loss(fake_outputs, fake_labels)
        d_loss = (d_loss_real + d_loss_fake) / 2
        d_loss.backward()
        discriminator_optimizer.step()

        # # 加密网络训练
        adv_loss = adversarial_loss(model.discriminator(encrypted_image), real_labels)
        encrypt_loss = MSE_loss(encrypted_image, key)    # 是不是应该把sharp改为key？
        decrypt_loss_l1 = L1_loss(decrypted_image, sharp)
        perceptual_loss_value = perceptual_loss(decrypted_image, sharp)
        key_loss = key_loss_func(sharp, decrypted_image, wrong_key)
        # hist_ent_loss = histogram_entropy_loss(encrypted_image, sharp)        # 直方图和熵损失

        # 计算总损失
        # total_loss = (lambda_mse * encrypt_loss + lambda_l1 * decrypt_loss_l1 + lambda_hist_ent * hist_ent_loss +
        #               lambda_perceptual * perceptual_loss_value - lambda_key * key_loss + adv_loss +
        #               lambda_diffusion * diff_loss + lambda_correlation * cor_loss + lambda_similarity * sim_loss +
        #               lambda_error * err_loss)

        total_loss = (lambda_mse * encrypt_loss + lambda_l1 * decrypt_loss_l1 +
                      lambda_perceptual * perceptual_loss_value - lambda_key * key_loss + adv_loss)

        # 反向传播和优化
        # optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        torch.cuda.empty_cache()

        # 记录各项损失
        train_hist['d_loss_real'].append(d_loss_real.item())
        train_hist['d_loss_fake'].append(d_loss_fake.item())
        train_hist['adv_loss'].append(adv_loss.item())
        train_hist['E_losses'].append(encrypt_loss.item())
        train_hist['D_losses_L1'].append(decrypt_loss_l1.item())
        train_hist['total_losses'].append(total_loss.item())

        # 记录损失到 TensorBoard
        # writer.add_scalar('Loss/d_loss_real', d_loss_real.item(), epoch * train_data_size + i)
        # writer.add_scalar('Loss/d_loss_fake', d_loss_fake.item(), epoch * train_data_size + i)
        # writer.add_scalar('Loss/adv_loss', adv_loss.item(), epoch * train_data_size + i)
        # writer.add_scalar('Loss/encrypt_loss', encrypt_loss.item(), epoch * train_data_size + i)
        # # writer.add_scalar('Loss/Encrypt', encrypt_loss.item(), epoch * train_data_size + len(train_hist['E_losses']))
        # writer.add_scalar('Loss/Decrypt_L1', decrypt_loss_l1.item(),
        #                   epoch * train_data_size + len(train_hist['D_losses_L1']))
        # writer.add_scalar('Loss/Total', total_loss.item(), epoch * train_data_size + len(train_hist['total_losses']))
        step = epoch * len(training_loader) + len(train_hist['E_losses'])
        writer.add_scalar('Loss/Encrypt_Loss', encrypt_loss.item(), step)
        writer.add_scalar('Loss/Decrypt_L1_Loss', decrypt_loss_l1.item(), step)
        writer.add_scalar('Loss/Discriminator_Real_Loss', d_loss_real.item(), step)
        writer.add_scalar('Loss/Discriminator_Fake_Loss', d_loss_fake.item(), step)
        writer.add_scalar('Loss/Adversarial_Loss', adv_loss.item(), step)
        writer.add_scalar('Loss/Total_Loss', total_loss.item(), step)


    if (epoch + 1) % valid_epoch == 0:
        val_sharp, val_blur = next(iter(validation_loader))
        with torch.no_grad():
            val_sharp = val_sharp.to(device)
            val_key = torch.ones_like(val_sharp)
            val_encrypted_A, val_decrypted = model(val_sharp, val_key)

        save_images(val_encrypted_A[0].unsqueeze(0), path=save_path, batch=epoch + 1, isGT='en_1')
        save_images(val_decrypted[0].unsqueeze(0), path=save_path, batch=epoch + 1, isGT='de_1')
        save_images(val_sharp[0].unsqueeze(0), path=save_path, batch=epoch + 1, isGT='Yuan_1')

        valid_originals = np.moveaxis(val_sharp.cpu().numpy(), 1, -1)  # 源代码是1，-1
        valid_reconstructions = np.moveaxis(val_decrypted.detach().cpu().numpy(), 1, -1)

        # 动态调整 win_size
        min_size = min(valid_originals.shape[1:3])  # 获取图像的最小尺寸
        win_size = min(7, min_size)  # 确保 win_size 不超过图像的最小尺寸且至少为7

        # 计算 PSNR 和 SSIM
        valid_psnr = 0.0
        valid_ssim = 0.0
        for i in range(valid_originals.shape[0]):
            # For color images, set multichannel=True
            valid_psnr += psnr(valid_originals[i], valid_reconstructions[i], data_range=1.0)
            # valid_ssim += ssim(valid_originals[i], valid_reconstructions[i], data_range=1.0, multichannel=True)  # 源代码
            valid_ssim += ssim(valid_originals[i], valid_reconstructions[i], data_range=1.0, win_size=win_size,
                               channel_axis=-1)

        print(
            f"Epoch {epoch + 1},  PSNR: {valid_psnr / valid_originals.shape[0]:.4f}, SSIM: {valid_ssim / valid_originals.shape[0]:.4f}")

    elif (epoch + 1) % print_epoch == 0:
        print(f'Epoch {epoch + 1}, ', end='')
        for key, value in train_hist.items():
            if key.endswith('_losses'):
                # Calculate the mean of the losses excluding the last print_epoch values
                mean_loss = torch.mean(torch.FloatTensor(value[:-print_epoch])).item()
                print(f'{key[:-7]}_loss: {mean_loss:.3f}, ', end='')
        print()


def train(training_loader, validation_loader, img_save_path, model_save_path, start_epoch=0):
    for epoch in range(start_epoch, num_epochs):
        train_epoch(epoch, training_loader, validation_loader, save_path=img_save_path)
        if (epoch + 1) % save_epoch == 0:
            torch.save(model.state_dict(), f'{model_save_path}/{checkpoints_name}_{str(epoch + 1)}.pth')

        # 保存损失值曲线
        plot_loss_curves(train_hist, model_save_path)
        # 关闭 SummaryWriter
        writer.close()


# # 测试
@torch.no_grad()
def test(data_loader, save_path):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = ImageCrypt(input_dim=1, num_filter=64, output_dim=1)
    model.to(device)
    # model.load_state_dict(torch.load(model_path))   ##测试的时候需要去掉注释
    model.eval()  # 进入评估模式
    data_size = len(data_loader)
    test_data = tqdm(data_loader, total=data_size, initial=1, leave=False)
    for index, img in enumerate(test_data):
        val_sharp = img[0].to(device)
        val_key = torch.ones_like(val_sharp)  # 正确的密钥
        wrong_key = torch.randint(0, 256, val_key.shape, dtype=torch.float).to(device) / 255.0  # 和正确密钥生成方式一样

        # 使用正确密钥解密
        encrypted_image_A, decrypted_image = model(val_sharp, val_key)
        # 使用错误的密钥解密
        _, val_decrypted_wrong = model(val_sharp, wrong_key)  # 接收所有返回值

        correct_save_path = os.path.join(save_path, r"correct_{index + 1}")
        wrong_save_path = os.path.join(save_path, r"wrong_{index + 1}")
        # 确保目录存在
        os.makedirs(correct_save_path, exist_ok=True)
        os.makedirs(wrong_save_path, exist_ok=True)
        print(f"Saving images to {correct_save_path} and {wrong_save_path}")

        # 保存解密结果
        save_images(val_sharp[0].unsqueeze(0), path=correct_save_path, batch=index + 1, isGT='gt')
        save_images(encrypted_image_A[0].unsqueeze(0), path=correct_save_path, batch=index + 1, isGT='en')
        save_images(decrypted_image[0].unsqueeze(0), path=correct_save_path, batch=index + 1, isGT='de_correct')
        save_images(val_decrypted_wrong[0].unsqueeze(0), path=wrong_save_path, batch=index + 1,
                    isGT='de_wrong')  # # 保存使用错误密钥的解密结果

        # 计算 PSNR 和 SSIM
        valid_originals = np.moveaxis(val_sharp.cpu().numpy(), 1, -1)
        valid_reconstructions = np.moveaxis(decrypted_image.detach().cpu().numpy(), 1, -1)
        valid_reconstructions_wrong = np.moveaxis(val_decrypted_wrong.detach().cpu().numpy(), 1, -1)

        # 动态调整 win_size
        min_size = min(valid_originals.shape[1:3])  # 获取图像的最小尺寸
        win_size = min(7, min_size)  # 确保 win_size 不超过图像的最小尺寸且至少为7

        valid_psnr_correct = psnr(valid_originals[0], valid_reconstructions[0], data_range=1.0)
        valid_psnr_wrong = psnr(valid_originals[0], valid_reconstructions_wrong[0], data_range=1.0)
        # valid_ssim_correct = ssim(valid_originals[0], valid_reconstructions[0], data_range=1.0, multichannel=True)
        # valid_ssim_wrong = ssim(valid_originals[0], valid_reconstructions_wrong[0], data_range=1.0, multichannel=True)
        valid_ssim_correct = ssim(valid_originals[0], valid_reconstructions[0], data_range=1.0, win_size=win_size,
                                  channel_axis=-1)
        valid_ssim_wrong = ssim(valid_originals[0], valid_reconstructions_wrong[0], data_range=1.0, win_size=win_size,
                                channel_axis=-1)

        key_loss_value = key_loss_func(val_sharp, decrypted_image, val_decrypted_wrong).item()

        print(f"Image {index + 1}: Correct Key - PSNR: {valid_psnr_correct:.4f}, SSIM: {valid_ssim_correct:.4f}")
        print(f"Image {index + 1}: Wrong Key - PSNR: {valid_psnr_wrong:.4f}, SSIM: {valid_ssim_wrong:.4f}")
        print(f"KeyLoss: {key_loss_value:.4f}")

        # 计算 PSNR 和 SSIM
        # valid_psnr = 0.0
        # valid_ssim = 0.0
        # for i in range(valid_originals.shape[0]):
        #
        #     valid_psnr += psnr(valid_originals[i], valid_reconstructions[i], data_range=1.0)
        #     valid_ssim += ssim(valid_originals[i], valid_reconstructions[i], data_range=1.0, multichannel=True)
        # print(
        #     f"Epoch {index + 1},  PSNR: {valid_psnr / valid_originals.shape[0]:.4f}, SSIM: {valid_ssim / valid_originals.shape[0]:.4f}")


if __name__ == '__main__':
    name = checkpoints_name + '_' + train_data_path.split('\\')[-2]
    model_save_path = f'./checkpoints/{name}'
    img_save_path = f'{model_save_path}/Data-picture'
    makeDir(model_save_path)
    makeDir(img_save_path)

    ##### 训练
    # training_loader, validation_loader, full_loader = GetDataLoader(train_data_path, image_size, batch_size=batch_size)
    # train(training_loader, validation_loader, img_save_path, model_save_path, start_epoch=0)
    # 训练数据加载器
    training_loader, validation_loader, full_loader = GetDataLoader(train_data_path, image_size, batch_size=batch_size)

    # 初始化判别器优化器
    discriminator_optimizer = optim.Adam(model.discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    train(training_loader, validation_loader, img_save_path, model_save_path, start_epoch=0)

    # # 训练和验证模型
    # for epoch in range(num_epochs):
    #     train_epoch(epoch, training_loader, validation_loader, img_save_path)
    #
    #     if (epoch + 1) % save_epoch == 0:
    #         torch.save(model.state_dict(), os.path.join(model_save_path, f'model_epoch_{epoch + 1}.pth'))
    #         print(f'Model saved at epoch {epoch + 1}')
    #
    # print('Finished Training')



    # # # # 测试
    # # model_path = r'checkpoints/demo_DATA/demo_5.pth'
    # model_path = r'checkpoints/Unet-base_DATA/Unet-base_15.pth'
    # model = torch.load(model_path)
    # # model.load_state_dict(torch.load('model_path.pth'))  # 导入网络的参数
    # test_path = r'F:\DATA\Tupian'
    # model_name = model_path.split("\\")[-1][:-4].split('_')[-1]
    # data_name = test_path.split("\\")[-2]
    # test_save_path = f'{model_save_path}/result/epoch={model_name}/{data_name}'
    # makeDir(test_save_path)
    #
    # training_loader, validation_loader,full_loader = GetDataLoader(train_data_path,image_size, test_path, batch_size=1)
    # test(full_loader, save_path=test_save_path)
