import torch
import torch.nn as nn
import torchvision.models as models

# ==========================================
# 1. CUSTOM ATTENTION BOTTLENECK (CBAM)
# ==========================================
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

class CBAM(nn.Module):
    def __init__(self, in_planes, ratio=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.ca = ChannelAttention(in_planes, ratio)
        self.sa = SpatialAttention(kernel_size)

    def forward(self, x):
        x = x * self.ca(x)
        x = x * self.sa(x)
        return x

# ==========================================
# 2. FEATURE AGGREGATION (SPP BLOCK)
# ==========================================
class SPPBlock(nn.Module):
    """
    Spatial Pyramid Pooling block to aggregate features across different scales,
    crucial for capturing varying sizes of infrastructure defects.
    """
    def __init__(self, in_channels, pool_sizes=(5, 9, 13)):
        super(SPPBlock, self).__init__()
        self.pools = nn.ModuleList([nn.MaxPool2d(kernel_size=ks, stride=1, padding=ks//2) for ks in pool_sizes])
        # The output channels will be in_channels * 4 (original + 3 pooled)
        self.conv = nn.Conv2d(in_channels * 4, in_channels, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(in_channels)
        self.relu = nn.LeakyReLU(0.1)

    def forward(self, x):
        features = [x]
        for pool in self.pools:
            features.append(pool(x))
        x = torch.cat(features, dim=1)
        x = self.relu(self.bn(self.conv(x)))
        return x

# ==========================================
# 3. THE FINAL ARCHITECTURE
# ==========================================
class UAVDefectDetector(nn.Module):
    def __init__(self, num_classes=3): # e.g., 0: fracture, 1: corrosion, 2: missing fastener
        super(UAVDefectDetector, self).__init__()
        
        # Load a validated backbone (ResNet50) and strip the final classification layers
        resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        self.backbone = nn.Sequential(*list(resnet.children())[:-2]) 
        
        # Output of ResNet50 before pooling is 2048 channels
        in_channels = 2048
        
        # Inject the custom extensions required by the rubric
        self.attention_bottleneck = CBAM(in_channels)
        self.spp = SPPBlock(in_channels)
        
        # Multi-task prediction heads (simultaneously predicting categories and boxes)
        self.class_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(in_channels, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )
        
        self.box_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(in_channels, 512),
            nn.ReLU(),
            nn.Linear(512, 4),
            nn.Sigmoid() # ADD THIS: Forces coordinate predictions between 0.0 and 1.0
        )

    def forward(self, x):
        # 1. Extract features through backbone
        features = self.backbone(x)
        
        # 2. Refine features to focus on small defects (Channel/Spatial Attention)
        refined_features = self.attention_bottleneck(features)
        
        # 3. Aggregate multi-scale context
        aggregated_features = self.spp(refined_features)
        
        # 4. Route to dual heads
        class_preds = self.class_head(aggregated_features)
        box_preds = self.box_head(aggregated_features)
        
        return class_preds, box_preds
