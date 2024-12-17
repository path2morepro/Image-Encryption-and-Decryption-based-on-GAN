from __future__ import print_function


import torch
import torch.nn as nn
import torch.nn.functional as F


# 使用正态分布初始化嵌入层的权重和EMA权重参数
class VectorQuantizerEMA(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, commitment_cost, decay, epsilon=1e-5):
        """
                初始化VectorQuantizerEMA类。
                VectorQuantizerEMA是用于执行矢量量化的一种模型，它使用指数移动平均（EMA）来更新嵌入向量。
                这种方法在生成式对抗网络（GANs）中，特别是用于图像生成的VQ-VAE模型中很常见。
                参数:
                - num_embeddings: int, 嵌入层中嵌入向量的数量。
                - embedding_dim: int, 每个嵌入向量的维度。
                - commitment_cost: float, 用于平衡编码器和解码器损失的因子。
                - decay: float, EMA更新的衰减率。
                - epsilon: float, 用于防止除以零错误的小数。
                """
        super(VectorQuantizerEMA, self).__init__()

        self._ema_cluster_size = None  # x新加
        self._embedding_dim = embedding_dim
        self._num_embeddings = num_embeddings

        # 初始化嵌入层，用于存储嵌入向量
        self._embedding = nn.Embedding(self._num_embeddings, self._embedding_dim)
        # 以均值为0，标准差为1的正态分布初始化嵌入层的权重
        self._embedding.weight.data.normal_()
        self._commitment_cost = commitment_cost

        self.register_buffer('_ema_cluster_size', torch.zeros(num_embeddings))
        # 初始化EMA权重参数，这将用于存储更新后的嵌入向量。
        self._ema_w = nn.Parameter(torch.Tensor(num_embeddings, self._embedding_dim))
        self._ema_w.data.normal_()

        self._decay = decay
        self._epsilon = epsilon

    def forward(self, inputs):
        """
               执行向量量化变分自编码器（VQ-VAE）的前向传播。
               参数:
               - inputs: 输入图像张量，形状为 (batch_size, channels, height, width)。
               返回:
               - loss: 需要最小化的损失值。
               - quantized: 量化后的输入张量，格式调整为 (batch_size, channels, height, width)。
               - perplexity: 表示编码分布的困惑度。
               - indices: 最近邻编码的索引。
               """
        # 将输入从 BCHW 格式转换为 BHWC 格式
        inputs = inputs.permute(0, 2, 3, 1).contiguous()
        input_shape = inputs.shape

        # 展平输入张量
        flat_input = inputs.view(-1, self._embedding_dim)

        # 计算输入与嵌入权重之间的距离
        distances = (torch.sum(flat_input ** 2, dim=1, keepdim=True)
                     + torch.sum(self._embedding.weight ** 2, dim=1)
                     - 2 * torch.matmul(flat_input, self._embedding.weight.t()))

        # 编码：找到距离最近的嵌入
        indices = torch.argmin(distances, dim=1)
        encoding_indices = indices.unsqueeze(1)
        encodings = torch.zeros(encoding_indices.shape[0], self._num_embeddings, device=inputs.device)
        encodings.scatter_(1, encoding_indices, 1)

        # 量化并重塑
        indices2quantized = torch.matmul(encodings, self._embedding.weight)
        quantized = indices2quantized.view(input_shape)

        # 使用指数移动平均更新嵌入向量
        if self.training:
            # 更新聚类大小的EMA
            self._ema_cluster_size = self._ema_cluster_size * self._decay + \
                                     (1 - self._decay) * torch.sum(encodings, 0)

            # 对聚类大小进行拉普拉斯平滑
            n = torch.sum(self._ema_cluster_size.data)
            self._ema_cluster_size = (
                    (self._ema_cluster_size + self._epsilon)
                    / (n + self._num_embeddings * self._epsilon) * n)

            # 更新权重的EMA
            dw = torch.matmul(encodings.t(), flat_input)
            self._ema_w = nn.Parameter(self._ema_w * self._decay + (1 - self._decay) * dw)

            # 更新嵌入权重
            self._embedding.weight = nn.Parameter(self._ema_w / self._ema_cluster_size.unsqueeze(1))

        # Loss损失计算
        e_latent_loss = F.mse_loss(quantized.detach(), inputs)
        loss = self._commitment_cost * e_latent_loss

        # Straight Through Estimator
        quantized = inputs + (quantized - inputs).detach()
        # 计算困惑度
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

        # 将量化后的张量格式从 BHWC 转换回 BCHW
        return loss, quantized.permute(0, 3, 1, 2).contiguous(), perplexity, indices


class Residual(nn.Module):
    def __init__(self, in_channels, num_hiddens, num_residual_hiddens):
        """  in_channels：输入通道数。
             num_hiddens：隐藏层的单元数。
            num_residual_hiddens：残差块中隐藏层的单元数。"""
        super(Residual, self).__init__()
        # 初始化一个序列模型，包含两个ReLU激活函数和两个卷积层。
        # 第一个卷积层用于扩展通道数，第二个卷积层用于减少通道数到输出所需的数量。
        self._block = nn.Sequential(
            nn.ReLU(True),
            nn.Conv2d(in_channels=in_channels,
                      out_channels=num_residual_hiddens,
                      kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(True),
            nn.Conv2d(in_channels=num_residual_hiddens,
                      out_channels=num_hiddens,
                      kernel_size=1, stride=1, bias=False)
        )

    def forward(self, x):
        return x + self._block(x)


class ResidualStack(nn.Module):
    def __init__(self, in_channels, num_hiddens, num_residual_layers, num_residual_hiddens):
        """
               初始化ResidualStack类的实例。
               这个类代表了一堆残差层（Residual Layers），用于深度学习模型中。残差层允许网络通过跳跃连接来学习更复杂的模式。
               参数:
               in_channels (int): 输入通道的数量，对应于卷积神经网络中的输入图像的通道数。
               num_hiddens (int): 每个残差层中隐藏单元的数量，决定了层的容量。
               num_residual_layers (int): 残差层的数量，决定了这个堆栈的深度。
               num_residual_hiddens (int): 每个残差层中隐藏单元的数量，与num_hiddens参数相同，用于指定每个残差块的大小。
               """
        super(ResidualStack, self).__init__()
        self._num_residual_layers = num_residual_layers
        # 初始化一个模块列表，其中包含num_residual_layers个残差层模块。
        # 每个残差层都由Residual类实例化，参数包括输入通道数、隐藏单元数和残差层内的隐藏单元数
        self._layers = nn.ModuleList([Residual(in_channels, num_hiddens, num_residual_hiddens)
                                      for _ in range(self._num_residual_layers)])

    def forward(self, x):
        for i in range(self._num_residual_layers):
            x = self._layers[i](x)
        return F.relu(x)


class VQ_VAE_Encoder(nn.Module):
    def __init__(self, in_channels, num_hiddens, num_residual_layers, num_residual_hiddens):
        """
                初始化编码器类。
                参数:
                in_channels (int): 输入通道数。
                num_hiddens (int): 隐藏层的单元数。
                num_residual_layers (int): 剩余堆栈中的层数。
                num_residual_hiddens (int): 剩余层中的隐藏单元数。
                """
        super(VQ_VAE_Encoder, self).__init__()

        # 初始化第一个卷积层，用于减少输入通道数并增大特征图的深度
        self._conv_1 = nn.Conv2d(in_channels=1,
                                 out_channels=num_hiddens // 2,
                                 kernel_size=4,
                                 stride=2, padding=1)
        self._conv_2 = nn.Conv2d(in_channels=num_hiddens // 2,
                                 out_channels=num_hiddens,
                                 kernel_size=4,
                                 stride=2, padding=1)
        self._conv_3 = nn.Conv2d(in_channels=num_hiddens,
                                 out_channels=num_hiddens,
                                 kernel_size=3,
                                 stride=1, padding=1)
        # 初始化剩余堆栈，这是编码器的主要组成部分，用于处理高级特征
        self._residual_stack = ResidualStack(in_channels=num_hiddens,
                                             num_hiddens=num_hiddens,
                                             num_residual_layers=num_residual_layers,
                                             num_residual_hiddens=num_residual_hiddens)

    def forward(self, inputs):
        x = self._conv_1(inputs)
        x = F.relu(x)

        x = self._conv_2(x)
        x = F.relu(x)

        x = self._conv_3(x)
        return self._residual_stack(x)        # 编码器输出的隐藏表示张量。


class VQ_VAE_Decoder(nn.Module):
    def __init__(self, embedding_dim, num_hiddens, num_residual_layers, num_residual_hiddens):
        super(VQ_VAE_Decoder, self).__init__()
        """
                初始化解码器类。
                参数:
                in_channels (int): 输入通道数，应与编码器输出的嵌入维度相同。
                num_hiddens (int): 隐藏层的单元数。
                num_residual_layers (int): 剩余堆栈中的层数。
                num_residual_hiddens (int): 剩余层中的隐藏单元数。
                """

        self._conv_1 = nn.Conv2d(in_channels=embedding_dim,
                                 out_channels=num_hiddens,
                                 kernel_size=3,
                                 stride=1, padding=1)

        self._residual_stack = ResidualStack(in_channels=num_hiddens,
                                             num_hiddens=num_hiddens,
                                             num_residual_layers=num_residual_layers,
                                             num_residual_hiddens=num_residual_hiddens)

        self._conv_trans_1 = nn.ConvTranspose2d(in_channels=num_hiddens,
                                                out_channels=num_hiddens // 2,
                                                kernel_size=4,
                                                stride=2, padding=1)

        self._conv_trans_2 = nn.ConvTranspose2d(in_channels=num_hiddens // 2,
                                                out_channels=3,
                                                kernel_size=4,
                                                stride=2, padding=1)

    def forward(self, inputs):
        x = self._conv_1(inputs)

        x = self._residual_stack(x)

        x = self._conv_trans_1(x)
        x = F.relu(x)
        # 通过转置卷积层 (self._conv_trans_1、self._conv_trans_2) 将处理后的张量还原为最终的重建图像张量
        return self._conv_trans_2(x)       # 解码器输出的重建图像张量


class Model(nn.Module):
    """
        初始化编码器 (self._encoder)、解码器 (self._decoder) 和量化器 (self._vq_vae)。
        通过 VectorQuantizerEMA 类定义的参数设置量化器。
         通过 Encoder 类和 Decoder 类定义的参数设置编码器和解码器。

        参数:
        - num_hiddens: 隐藏层神经元的数量。
        - num_residual_layers: ResNet块的数量。
        - num_residual_hiddens: ResNet块中隐藏层神经元的数量。
        - num_embeddings: 离散嵌入的数量。
        - embedding_dim: 嵌入维度。
        - commitment_cost: 用于平衡编码器和解码器损失的承诺成本。
        - decay: EMA（指数移动平均）的衰减率。
        """
    def __init__(self, num_hiddens, num_residual_layers, num_residual_hiddens,
                 num_embeddings, embedding_dim, commitment_cost, decay=0):
        super(Model, self).__init__()

        # 初始化一个编码器，用于将输入数据转换为隐藏表示。
        self._encoder = VQ_VAE_Encoder(1, num_hiddens, num_residual_layers, num_residual_hiddens)

        # 初始化一个卷积层，用于将编码器的输出转换为适合矢量量化的维度。
        self._pre_vq_conv = nn.Conv2d(in_channels=num_hiddens, out_channels=embedding_dim, kernel_size=1, stride=1)

        # 初始化矢量量化模块，用于将连续的隐藏表示离散化为嵌入向量。
        self._vq_vae = VectorQuantizerEMA(num_embeddings, embedding_dim, commitment_cost, decay)

        # 初始化一个解码器，用于从离散的嵌入向量重建输入数据
        self._decoder = VQ_VAE_Decoder(embedding_dim, num_hiddens, num_residual_layers, num_residual_hiddens)

        self.noise = None

    def forward(self, x):
        """
                前向传播函数，负责执行模型的前向计算。
                输入一个图像x，通过编码器、预VQ卷积、向量量化变分自编码器（VQ-VAE）和解码器，
                计算重建损失、获取量化后的表示、计算困惑度，并返回重建图像和中间变量。
                参数:
                x: 输入的图像Tensor。
                返回:
                loss: 重建损失Tensor。
                x_recon: 重建后的图像Tensor。
                perplexity: 困惑度Tensor，用于衡量编码器输出到词汇表的分布的均匀程度。
                z: 经过向量量化后的特征Tensor。
                """

        z = self._encoder(x)                   # 通过编码器获得特征z
        z = self._pre_vq_conv(z)
        # 初始化或更新噪声Tensor，用于增加训练的多样性
        if self.noise is None:
            self.noise = torch.randn_like(z) * 10

        z = z + self.noise                  # 将噪声添加到特征z中，以增加模型的泛化能力

        # 通过VQ-VAE模块，计算损失、量化后的特征、困惑度等
        loss, quantized, perplexity, _ = self._vq_vae(z)
        x_recon = self._decoder(quantized)        # 使用量化后的特征通过解码器获得重建图像

        return loss, x_recon, perplexity, z       # 返回损失、重建图像、困惑度和量化后的特征
