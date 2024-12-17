# losses.py
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
from numpy.testing import assert_almost_equal
# from torchvision import models
from torchvision.models import VGG16_Weights
import torch
from numpy.testing import assert_almost_equal





# 感知损失
class PerceptualLoss(nn.Module):
    def __init__(self, resize=True):
        super(PerceptualLoss, self).__init__()
        vgg16 = models.vgg16(weights=VGG16_Weights.IMAGENET1K_V1).features
        self.vgg_layers = nn.Sequential(*list(vgg16)[:16]).eval()
        self.resize = resize

        for param in self.vgg_layers.parameters():
            param.requires_grad = False

    def forward(self, input, target):
        if input.size(1) == 1:
            input = input.repeat(1, 3, 1, 1)
        if target.size(1) == 1:
            target = target.repeat(1, 3, 1, 1)

        if self.resize:
            input = nn.functional.interpolate(input, size=(224, 224), mode='bilinear', align_corners=False)
            target = nn.functional.interpolate(target, size=(224, 224), mode='bilinear', align_corners=False)

        input_features = self.vgg_layers(input)
        target_features = self.vgg_layers(target)

        loss = nn.functional.mse_loss(input_features, target_features)
        return loss


# 密钥复合损失
class KeyLoss(nn.Module):
    def __init__(self):
        super(KeyLoss, self).__init__()
        self.loss = nn.MSELoss()

    def forward(self, original, decrypted_img, wrong_key):
        correct_loss = self.loss(decrypted_img, original)
        wrong_loss = self.loss(wrong_key, original)
        return correct_loss - wrong_loss  # 直接将 wrong_loss 取负，以实现最大化它的效果



class CorrelationLoss(nn.Module):
    def __init__(self, lambda_cor=1.0, lambda_2d=1.0, num_bins=8):
        super(CorrelationLoss, self).__init__()
        self.lambda_cor = lambda_cor
        self.lambda_2d = lambda_2d
        self.num_bins = num_bins

    def forward(self, x):
        cor_loss = self.compute_correlation_loss(x)
        hist_2d_loss = self.compute_2d_histogram_loss(x)
        return self.lambda_cor * cor_loss + self.lambda_2d * hist_2d_loss

    def compute_correlation_loss(self, x):
        N, C, H, W = x.size()
        x_flat = x.view(N * C, -1)
        y_flat = torch.cat([x_flat[:, 1:], x_flat[:, :1]], dim=1)
        x_mean = x_flat.mean(dim=1, keepdim=True)
        y_mean = y_flat.mean(dim=1, keepdim=True)
        cov_xy = ((x_flat - x_mean) * (y_flat - y_mean)).mean(dim=1)

        var_x = ((x_flat - x_mean) ** 2).mean(dim=1)
        var_y = ((y_flat - y_mean) ** 2).mean(dim=1)
        cor_xy = cov_xy / (torch.sqrt(var_x) * torch.sqrt(var_y) + 1e-10)
        cor_loss = torch.mean(torch.abs(cor_xy))
        return cor_loss

    def compute_2d_histogram_loss(self, x):
        N, C, H, W = x.size()
        x_flat = x.view(N * C, -1).detach().cpu().numpy().flatten()
        y_flat = np.roll(x_flat, shift=-1)
        hist_2d, _, _ = np.histogram2d(x_flat, y_flat, bins=self.num_bins, range=[[0, 1], [0, 1]])
        hist_2d = hist_2d / hist_2d.sum()
        ideal_hist_2d = np.ones_like(hist_2d) / (self.num_bins * self.num_bins)
        hist_2d_loss = np.mean((hist_2d - ideal_hist_2d) ** 2)
        hist_2d_loss = torch.tensor(hist_2d_loss, device=x.device)
        return hist_2d_loss
#
#
# 相似损失
class SimilarityLoss(nn.Module):
    def __init__(self, epsilon=0, lambda_dissim=1.0, lambda_sim=1.0):
        super(SimilarityLoss, self).__init__()
        self.epsilon = epsilon
        self.lambda_dissim = lambda_dissim
        self.lambda_sim = lambda_sim

    def forward(self, alpha, inne_alpha, ep, innr_ep):
        dissim_loss = self.compute_dissim_loss(alpha, inne_alpha)
        sim_loss = self.compute_sim_loss(alpha, innr_ep)
        return self.lambda_dissim * dissim_loss + self.lambda_sim * sim_loss

    def compute_dissim_loss(self, alpha, inne_alpha):
        cos_sim = nn.functional.cosine_similarity(alpha, inne_alpha, dim=-1)
        dissim_loss = torch.max(torch.tensor(self.epsilon, device=cos_sim.device), cos_sim).mean()
        return dissim_loss

    def compute_sim_loss(self, alpha, innr_ep):
        cos_sim = nn.functional.cosine_similarity(alpha, innr_ep, dim=-1)
        sim_loss = (1 - cos_sim).mean()
        return sim_loss



