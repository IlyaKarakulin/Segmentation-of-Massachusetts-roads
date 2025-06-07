import torch
import torch.nn as nn
import torch.nn.functional as F


#! ------ Single Lavel Encoder ------

class DilatedConvBlock(nn.Module):
    """Dilated Convolution Block with residual connection"""
    def __init__(self, num_channels, dilation_rate):
        super(DilatedConvBlock, self).__init__()
        
        self.conv1x1_1 = nn.Conv2d(num_channels, num_channels, 1, padding=0)
        self.conv3x3_1 = nn.Conv2d(num_channels, num_channels, 3, 
                                   dilation=dilation_rate, padding=dilation_rate)
        self.bn1 = nn.BatchNorm2d(num_channels)
        
        self.conv1x1_2 = nn.Conv2d(num_channels, num_channels, 1, padding=0)
        self.conv3x3_2 = nn.Conv2d(num_channels, num_channels, 3,
                                   dilation=dilation_rate, padding=dilation_rate)
        self.bn2 = nn.BatchNorm2d(num_channels)
        
        self.residual = nn.Conv2d(num_channels, num_channels, 1)
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        residual = self.residual(x)
        
        out = self.conv1x1_1(x)
        out = self.conv3x3_1(out)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv1x1_2(out)
        out = self.conv3x3_2(out)
        out = self.bn2(out)
        
        out += residual
        out = self.relu(out)
        
        return out


class DPMG(nn.Module):
    """Dynamic Probability Map Generator"""

    def __init__(self, in_channels):
        super(DPMG, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels, in_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.relu1 = nn.ReLU(inplace=True)
        
        self.conv2 = nn.Conv2d(in_channels, in_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(in_channels)
        self.relu2 = nn.ReLU(inplace=True)
        
        self.conv3 = nn.Conv2d(in_channels, in_channels, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(in_channels)
        self.relu3 = nn.ReLU(inplace=True)
        
        self.final_conv = nn.Conv2d(in_channels, 1, 1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))
        x = self.relu3(self.bn3(self.conv3(x)))
        x = self.final_conv(x)
        # x = self.sigmoid(x)
        return x


class FeatureEncoder(nn.Module):
    """Feature Encoder"""

    def __init__(self, num_channels, path_index):
        super(FeatureEncoder, self).__init__()
        
        self.path_index = path_index
        
        self.dilated_blocks = nn.ModuleList()
        
        for j in range(1, 5):  # j = 1, 2, 3, 4
            dilation_rate = path_index * j  # i.j.r где r=1
            self.dilated_blocks.append(
                DilatedConvBlock(num_channels, dilation_rate)
            )
            
    def forward(self, x):
        for block in self.dilated_blocks:
            x = block(x)
        return x


class SingleLevelEncoder(nn.Module):
    """Single Level Encoder"""

    def __init__(self, num_channels, path_index):
        super(SingleLevelEncoder, self).__init__()
        
        self.feature_encoder = FeatureEncoder(num_channels, path_index)
        self.dpmg = DPMG(num_channels)
        
    def forward(self, x):
        features = self.feature_encoder(x)  # fi
        prob_map = self.dpmg(features)      # mi
        return features, prob_map


#! ------ Dymamic Attention Map Guided Index Pooling (DAMIP) ------

class IndexPooling(nn.Module):
    """Index Pooling layer"""
    def __init__(self, kernel_size=2):
        super(IndexPooling, self).__init__()
        self.kernel_size = kernel_size
        
    def forward(self, x):
        B, C, H, W = x.shape
        k = self.kernel_size
        
        output_maps = []
        
        for i in range(k):
            for j in range(k):
                map_ij = x[:, :, i::k, j::k]
                output_maps.append(map_ij)
        
        result = torch.cat(output_maps, dim=1)  # (B, C*k^2, H//k, W//k)
        
        return result

class DAMIP(nn.Module):
    """Dynamic Attention Map guided Index Pooling"""
    def __init__(self, in_channels, out_channels):
        super(DAMIP, self).__init__()
        self.index_pooling = IndexPooling(kernel_size=2)
        self.bottleneck = nn.Conv2d(in_channels * 4, out_channels, 1)
        self.final_conv = nn.Conv2d(out_channels, out_channels, 7, padding=3)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, feature_map, prob_map):

        pooled_features = self.index_pooling(feature_map)  # (B, C*4, H/2, W/2)
        pooled_probs = self.index_pooling(prob_map)        # (B, 4, H/2, W/2)
        
        B, C_feat, H, W = pooled_features.shape
        original_channels = C_feat // 4
        
        pooled_probs_expanded = pooled_probs.repeat(1, original_channels, 1, 1)
        
        attention_features = pooled_features * pooled_probs_expanded + pooled_features
        
        out = self.bottleneck(attention_features)
        out = self.final_conv(out)
        out = self.bn(out)
        out = self.relu(out)
        
        return out



#! ------ Dynamic Attention Map guided Spatial and Channel Attention (DAMSCA)------

class DAMSCA(nn.Module):
    """Dynamic Attention Map guided Spatial and Channel Attention"""
    def __init__(self, in_channels):
        super(DAMSCA, self).__init__()
        
        self.channel_conv = nn.Conv2d(1, in_channels, 1)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.sigmoid = nn.Sigmoid()
        
        self.relu = nn.ReLU(inplace=True)
        
        self.upsample = nn.Upsample(size=(256, 256), mode='bilinear', align_corners=False)
        
    def forward(self, features, prob_map):
        if prob_map.shape[2:] != features.shape[2:]:
            prob_map = F.interpolate(prob_map, size=features.shape[2:], mode='bilinear', align_corners=False)
        
        spatial_attention = features * prob_map
        
        channel_prob = self.channel_conv(prob_map)
        channel_weights = self.gap(channel_prob)
        channel_weights = self.sigmoid(channel_weights)  # [0, 1]
        channel_attention = features * channel_weights
        
        combined = spatial_attention + channel_attention
        combined = self.relu(combined)
        
        upsampled = self.upsample(combined)
        
        return upsampled


#! ----- Decoder ------

class Decoder(nn.Module):
    """Decoder для генерации финальной карты сегментации"""
    def __init__(self, in_channels):
        super(Decoder, self).__init__()
        
        self.transpose_conv = nn.ConvTranspose2d(in_channels, in_channels // 8, kernel_size=4, stride=2, padding=1)
        
        self.conv1 = nn.Conv2d(in_channels // 8, in_channels // 16, 3, padding=1)
        self.conv2 = nn.Conv2d(in_channels // 16, in_channels // 32, 3, padding=1)
        self.conv3 = nn.Conv2d(in_channels // 32, 1, 3, padding=1)
        
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        x = self.transpose_conv(x)  # H/2×W/2 -> H×W, 15C -> 2C
        x = self.conv1(x)           # 2C -> C
        x = self.conv2(x)           # C -> C/2
        x = self.conv3(x)           # C/2 -> 1
        # x = self.sigmoid(x)         # [0, 1]
        return x
    

#! ------ Multi-Scale Supervised Dilated Multiple-Path Attention Network -----

class MSSDMPA_Net(nn.Module):
    def __init__(self, input_channels=3, base_channels=64):
        super(MSSDMPA_Net, self).__init__()
        self.base_channels = base_channels
        
        self.input_conv = nn.Conv2d(input_channels, base_channels, 7, stride=2, padding=3)
        self.input_bn = nn.BatchNorm2d(base_channels)
        self.input_relu = nn.ReLU(inplace=True)
        
        path_channels = [base_channels, base_channels*2, base_channels*4, base_channels*8]
        
        self.encoders = nn.ModuleList()
        self.damip_modules = nn.ModuleList()
        self.damsca_modules = nn.ModuleList()
        
        for i in range(4):
            self.encoders.append(SingleLevelEncoder(path_channels[i], i+1))
            
            if i > 0:
                self.damip_modules.append(DAMIP(path_channels[i-1], path_channels[i]))
            
            self.damsca_modules.append(DAMSCA(path_channels[i]))
        
        total_channels = sum(path_channels)  # 15C
        self.decoder = Decoder(total_channels)
    
    def forward(self, x):
        prob_maps = []
        damip_outputs = []
        damsca_outputs = []
        
        g1 = self.input_relu(self.input_bn(self.input_conv(x)))
        
        for i in range(4):
            if i == 0:
                current_input = g1
            else:
                prev_prob_map = prob_maps[i-1]
                current_input = self.damip_modules[i-1](damip_outputs[i-1], prev_prob_map)
            
            damip_outputs.append(current_input)
            
            features, prob_map = self.encoders[i](current_input)
            prob_maps.append(prob_map)
            
            damsca_out = self.damsca_modules[i](features, prob_map)
            damsca_outputs.append(damsca_out)
        
        fdec = torch.cat(damsca_outputs, dim=1)
        mout = self.decoder(fdec)
        
        return mout, prob_maps[0], prob_maps[1], prob_maps[2], prob_maps[3]

    