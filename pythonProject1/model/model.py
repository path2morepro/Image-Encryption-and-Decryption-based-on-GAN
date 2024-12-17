import torch
import torch.nn as nn
import torch.nn.functional as F

class Convolution(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv1 = nn.Conv2d(in_c, out_c, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_c)
        self.conv2 = nn.Conv2d(out_c, out_c, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_c)
        self.relu = nn.ReLU()
    def forward(self, inputs):
        x = self.conv1(inputs)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        return x

class Encoder(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv = Convolution(in_c, out_c)
        self.pool = nn.MaxPool2d((2, 2))
    def forward(self, inputs):
        x = self.conv(inputs)
        p = self.pool(x)
        return x, p

class Decoder(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_c, out_c, kernel_size=2, stride=2)
        self.conv = Convolution(out_c + out_c, out_c)
    def forward(self, inputs, skip):
        x = self.up(inputs)
        x = F.interpolate(x, size=skip.size()[2:])
        x = torch.cat([x, skip], dim=1)
        x = self.conv(x)
        return x

class UNET(nn.Module):
    def __init__(self):
        super().__init__()
        self.e1 = Encoder(1, 64)
        self.e2 = Encoder(64, 128)
        self.e3 = Encoder(128, 256)
        self.e4 = Encoder(256, 512)
        self.b = Convolution(512, 1024)
        self.d1 = Decoder(1024, 512)
        self.d2 = Decoder(512, 256)
        self.d3 = Decoder(256, 128)
        self.d4 = Decoder(128, 64)
        self.outputs = nn.Conv2d(64, 1, kernel_size=1, padding=0)

    def forward(self, inputs):
        s1, p1 = self.e1(inputs)
        s2, p2 = self.e2(p1)
        s3, p3 = self.e3(p2)
        s4, p4 = self.e4(p3)

        b = self.b(p4)
        d1 = self.d1(b, s4)
        d2 = self.d2(d1, s3)
        d3 = self.d3(d2, s2)
        d4 = self.d4(d3, s1)

        outputs = self.outputs(d4)
        return outputs

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Flatten(),
            nn.Linear(256 * 32 * 32, 1),
            nn.Sigmoid()
        )

    def forward(self, img):
        return self.model(img)
    

    


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