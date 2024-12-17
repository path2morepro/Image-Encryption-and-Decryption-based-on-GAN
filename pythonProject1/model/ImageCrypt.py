import numpy as np
import torch
import torch.nn as nn

class ImageCrypt(torch.nn.Module):
    def __init__(self, input_dim=1, num_filter=64, output_dim=1):
        super(ImageCrypt, self).__init__()     # 调用父类的构造函数
        self.encryptor = Encoder(input_dim)
        self.generator = Decoder(input_dim * 2, num_filter, output_dim)
        self.discriminator = Discriminator(input_dim, num_filter)  # 添加判别器


    def forward(self, x, key):

        # 加密过程
        encrypted_input = torch.cat([x, key], dim=1)
        encrypted_image= self.encryptor(encrypted_input)

        # 解密过程
        decrypted_input = torch.cat([encrypted_image, key], dim=1)
        decrypted_image = self.generator(decrypted_input)

        return encrypted_image, decrypted_image

class Noise(torch.nn.Module):
    def __init__(self, input_dim):
        super(Noise, self).__init__()
        # encoder这边有个卷积是因为加上key以后channel会有变化1-2
        self.conv = ConvBlock(input_size=input_dim*2, output_size=input_dim, kernel_size=3, stride=1, padding=1)
    def forward(self, x):
        x = self.conv(x)
        device = x.device
        noise = torch.randn_like(x)
        noise = noise.to(device)
        output = x + noise
        return x, output
    
class Decoder(torch.nn.Module):
    def __init__(self, input_dim, num_filter, output_dim):
        super(Decoder, self).__init__()
         # Encoder
        self.conv1 = ConvBlock(input_dim, num_filter, activation=False, batch_norm=False)
        self.conv2 = ConvBlock(num_filter, num_filter * 2)
        self.conv3 = ConvBlock(num_filter * 2, num_filter * 4)
        self.conv4 = ConvBlock(num_filter * 4, num_filter * 8)
        self.conv5 = ConvBlock(num_filter * 8, num_filter * 16, batch_norm=False)

        # Decoder
        self.deconv1 = DeconvBlock(num_filter * 16, num_filter * 8)
        self.deconv2 = DeconvBlock(num_filter * 16, num_filter * 4)
        self.deconv3 = DeconvBlock(num_filter * 8, num_filter * 2)
        self.deconv4 = DeconvBlock(num_filter * 4, num_filter)
        self.deconv5 = DeconvBlock(num_filter * 2, output_dim, batch_norm=False)

        # Final convolution to reduce channel to 1
        self.final_conv1 = nn.Conv2d(output_dim, output_dim, kernel_size=3, padding=1)
        self.final_conv2 = nn.Conv2d(output_dim, 1, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Encoder
        enc1 = self.conv1(x)
        enc2 = self.conv2(enc1)
        enc3 = self.conv3(enc2)
        enc4 = self.conv4(enc3)
        enc5 = self.conv5(enc4)

        dec1 = self.deconv1(enc5, enc4)
        dec2 = self.deconv2(dec1, enc3)
        dec3 = self.deconv3(dec2, enc2)
        dec4 = self.deconv4(dec3, enc1)
        dec5 = self.deconv5(dec4)

        out = self.final_conv1(dec5)
        out = self.final_conv2(out)
        out = self.sigmoid(out)  # 使用非就地操作
        return out
    
    #decoder太菜了要改
        

class Discriminator(torch.nn.Module):
    def __init__(self, input_dim, num_filter):
        super(Discriminator, self).__init__()
        self.conv1 = ConvBlock(input_dim, num_filter, kernel_size=4, stride=2, padding=1)
        self.conv2 = ConvBlock(num_filter, num_filter * 2, kernel_size=4, stride=2, padding=1)
        self.conv3 = ConvBlock(num_filter * 2, num_filter * 4, kernel_size=4, stride=2, padding=1)
        self.conv4 = ConvBlock(num_filter * 4, num_filter * 8, kernel_size=4, stride=2, padding=1)
        self.final_conv = nn.Conv2d(num_filter * 8, 1, kernel_size=4, stride=1, padding=0)
        self.sigmoid = nn.Sigmoid()
        

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.final_conv(x)
        x = self.sigmoid(x)
        return x



   # 必须还原到原来尺寸
        # 经过这么一个encoder以后，decoder的输入要求是和图像大小一样的，并且要加上key
        # 所以最后的结果应该是图像encoder之后， 在经历encoder-decoder最终出现图像，所以算上encoder， decoder和encoder是两个独立的encoder-decoder结构
        # 所以这个generator其实是一个decoder，encoder的话，我们直接加密就行了
        # 所以encoder是一个加密层， decoder是一个encoder-decoder的编码器

class ConvBlock(torch.nn.Module):
    def __init__(self, input_size, output_size, kernel_size=4, stride=2, padding=1, activation=True, batch_norm=True):
        super(ConvBlock, self).__init__()
        self.conv = torch.nn.Conv2d(input_size, output_size, kernel_size, stride, padding)      # kernel_size卷积核大小，stride卷积步长，padding卷积填充的数量
        self.activation = activation
        self.lrelu = torch.nn.LeakyReLU(0.2, inplace=False)  # 设置 inplace=False
        self.batch_norm = batch_norm
        self.bn = torch.nn.BatchNorm2d(output_size)

      # 前向传播
    def forward(self, x):
        if self.activation:
            out = self.conv(self.lrelu(x))
        else:
            out = self.conv(x)

        if self.batch_norm:
            return self.bn(out)
        else:
            return out

class DeconvBlock(torch.nn.Module):
    def __init__(self, input_size, output_size, kernel_size=4, stride=2, padding=1, batch_norm=True, dropout=False, use_skip=True):
        super(DeconvBlock, self).__init__()
        self.use_skip = use_skip
        self.deconv = torch.nn.ConvTranspose2d(input_size, output_size, kernel_size, stride, padding)
        self.bn = torch.nn.BatchNorm2d(output_size)
        self.drop = torch.nn.Dropout(0.5 if dropout else 0)  # 增加dropout概率以防止过拟合
        self.relu = torch.nn.ReLU(inplace=False)  # 设置 inplace=False
        self.batch_norm = batch_norm
        self.dropout = dropout

    def forward(self, x, skip_input=None):
        x = self.deconv(self.relu(x))
        if self.batch_norm:
            x = self.bn(x)
        if self.dropout:
            x = self.drop(x)
        if self.use_skip and skip_input is not None:
            x = torch.cat([x, skip_input], 1)  # 将跳过连接的输入合并
        return x




