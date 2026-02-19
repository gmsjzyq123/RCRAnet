import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init


class ViTAttention(nn.Module):
    def __init__(self, dim, num_heads=4, depth=1, mlp_ratio=4):
        super(ViTAttention, self).__init__()
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=dim,  # embedding 维度
                nhead=num_heads,  # 多头注意力头数
                dim_feedforward=dim * mlp_ratio,  # FFN 维度
                dropout=0.1,
                batch_first=True
            ),
            num_layers=depth  # Transformer 层数
        )

    def forward(self, x):
        return self.transformer(x)


class RE(nn.Module):
    def __init__(self, input_channels, n_classes, patch_size=5, dilation=1, heads=2, encodes=1, mlp=4):
        super(RE, self).__init__()
        self.patch_size = patch_size
        self.input_channels = input_channels
        dilation = (dilation, 1, 1)

        # 第一层 3D 卷积
        self.conv1 = nn.Conv3d(1, 20, (3, 3, 3), stride=1, dilation=dilation, padding=0)
        self.pool1 = nn.Conv3d(20, 20, (3, 1, 1), dilation=dilation, stride=(2, 1, 1), padding=(1, 0, 0))

        # 多分支卷积
        self.conv2 = nn.ModuleList([
            nn.Conv3d(20, 35, (3, 1, 1), dilation=dilation, stride=1, padding=(1, 0, 0)),
            nn.Conv3d(20, 35, (1, 3, 1), dilation=dilation, stride=1, padding=(0, 1, 0)),
            nn.Conv3d(20, 35, (1, 1, 3), dilation=dilation, stride=1, padding=(0, 0, 1))
        ])
        self.Conv_mixnas_2 = nn.Conv3d(35 * 3, 35, kernel_size=1, stride=1, padding=0)
        self.pool2 = nn.Conv3d(35, 35, (3, 1, 1), dilation=dilation, stride=(2, 1, 1), padding=(1, 0, 0))

        self.conv3 = nn.Conv3d(35, 35, (3, 1, 1), dilation=dilation, stride=1, padding=(1, 0, 0))
        self.conv4 = nn.Conv3d(35, 35, (2, 1, 1), dilation=dilation, stride=(2, 1, 1), padding=(1, 0, 0))

        # ViT 注意力机制
        self.vit = ViTAttention(dim=input_channels, num_heads=heads, depth=encodes, mlp_ratio=mlp)
        # self.vit = ViTAttention(dim=input_channels, num_heads=2, depth=1, mlp_ratio=4)
        self.vit_conv = nn.Conv3d(in_channels=20, out_channels=35, kernel_size=(3, 1, 1), stride=(2, 1, 1),
                                  padding=(1, 0, 0))

        self.features_size = self._get_final_flattened_size()
        self.fc = nn.Linear(self.features_size, n_classes)
        self.pro_head = nn.Linear(self.features_size, 128)

    def _get_final_flattened_size(self):
        with torch.no_grad():
            x = torch.zeros((1, 1, self.input_channels, self.patch_size, self.patch_size))
            x = F.relu(self.conv1(x))
            x = self.pool1(x)

            spectralnas = []
            for layer in self.conv2:
                spectralnas.append(layer(x))
            spectralnas = torch.cat(spectralnas, dim=1)
            x = F.relu(self.Conv_mixnas_2(spectralnas))
            x = self.pool2(x)

            x = F.relu(self.conv3(x))
            x = F.relu(self.conv4(x))

            _, t, c, w, h = x.size()
        return t * c * w * h

    def forward(self, x, mode='use_f1'):
        # ViT 特征提取
        y = x  # [32,102,7,7]

        b, c, h, w = y.shape
        y = y.view(b, c, -1).permute(0, 2, 1)  # [B, P, D] # [32,49,102]

        y = self.vit(y)  # ViT 注意力增强 # [32,49,102]

        y = y.permute(0, 2, 1).view(b, c, h, w)  # 还原形状 # [32,102,7,7]

        y = y.unsqueeze(1)  # [32,1,102,7,7]
        y = F.relu(self.conv1(y))  # [32,20,100,5,5]

        # 三分支卷积特征提取
        x = x.unsqueeze(1)  # 增加通道维度 [B, 1, C, H, W]# [32,1,102,7,7]

        x = F.relu(self.conv1(x))  # [32,20,100,5,5]

        x = self.pool1(x)  # [32,20,50,5,5]

        spectralnas = []
        for layer in self.conv2:
            spectralnas.append(layer(x))
        spectralnas = torch.cat(spectralnas, dim=1)  # [32,105,50,5,5]

        x = F.relu(self.Conv_mixnas_2(spectralnas))  # [32,35,50,5,5]

        y = self.vit_conv(y)  # [32,35,50,5,5]

        x = x + y   # 无ViT

        x = self.pool2(x)  # [32,35,25,5,5]
        ## x = self.pool2(y)  # 无RIM

        x = F.relu(self.conv3(x))  # [32,35,25,5,5]

        x = F.relu(self.conv4(x))  # [32,35,13,5,5]

        if mode == 'use_f1':
            last_feat = x
            A = last_feat.reshape((last_feat.shape[0], last_feat.shape[1], -1))
            U, Sigma, VT = torch.svd(A)

            Sigma_F2 = torch.norm(Sigma, dim=1, keepdim=True)
            Sigma_F1_loss = torch.mean(torch.norm(Sigma / Sigma_F2, dim=1, p=1))
        else:
            Sigma_F1_loss = 0

        x = x.reshape(-1, self.features_size)
        x = self.fc(x)

        return (x, Sigma_F1_loss) if mode == 'use_f1' else x


def HamidaEtAl_RE(n_bands=1, n_classes=2, patch_size=1, heads=2, encodes=1, mlp=4):
    return RE(n_bands, n_classes, patch_size, heads=heads, encodes=encodes, mlp=mlp)
