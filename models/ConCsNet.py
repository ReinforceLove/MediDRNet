import torch
from torch import nn
import torch.nn.functional as F
from torchvision import models
# import torchsummary
from torchvision.models import ResNet18_Weights, ResNet50_Weights


# from loss.SupContrast import SupConLoss


class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True,
                 bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding,
                              dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes, eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class ChannelGate(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg', 'max']):
        super(ChannelGate, self).__init__()
        self.gate_channels = gate_channels
        self.mlp = nn.Sequential(
            Flatten(),
            nn.Linear(gate_channels, gate_channels // reduction_ratio),
            nn.ReLU(),
            nn.Linear(gate_channels // reduction_ratio, gate_channels)
        )
        self.pool_types = pool_types

    def forward(self, x):
        channel_att_sum = None
        for pool_type in self.pool_types:
            if pool_type == 'avg':
                avg_pool = F.avg_pool2d(x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp(avg_pool)
            elif pool_type == 'max':
                max_pool = F.max_pool2d(x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp(max_pool)
            elif pool_type == 'lp':
                lp_pool = F.lp_pool2d(x, 2, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp(lp_pool)
            elif pool_type == 'lse':
                # LSE pool only
                lse_pool = logsumexp_2d(x)
                channel_att_raw = self.mlp(lse_pool)

            if channel_att_sum is None:
                channel_att_sum = channel_att_raw
            else:
                channel_att_sum = channel_att_sum + channel_att_raw

        scale = F.sigmoid(channel_att_sum).unsqueeze(2).unsqueeze(3).expand_as(x)
        return x * scale


def logsumexp_2d(tensor):
    tensor_flatten = tensor.view(tensor.size(0), tensor.size(1), -1)
    s, _ = torch.max(tensor_flatten, dim=2, keepdim=True)
    outputs = s + (tensor_flatten - s).exp().sum(dim=2, keepdim=True).log()
    return outputs


class ChannelPool(nn.Module):
    def forward(self, x):
        return torch.cat((torch.max(x, 1)[0].unsqueeze(1), torch.mean(x, 1).unsqueeze(1)), dim=1)


class SpatialGate(nn.Module):
    def __init__(self):
        super(SpatialGate, self).__init__()
        kernel_size = 7
        self.compress = ChannelPool()
        self.spatial = BasicConv(2, 1, kernel_size, stride=1, padding=(kernel_size - 1) // 2, relu=False)

    def forward(self, x):
        x_compress = self.compress(x)
        x_out = self.spatial(x_compress)
        scale = F.sigmoid(x_out)  # broadcasting
        return x * scale


class CBAM(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg', 'max'], no_spatial=False):
        super(CBAM, self).__init__()
        self.ChannelGate = ChannelGate(gate_channels, reduction_ratio, pool_types)
        self.no_spatial = no_spatial
        if not no_spatial:
            self.SpatialGate = SpatialGate()

    def forward(self, x):
        x_out = self.ChannelGate(x)
        if not self.no_spatial:
            x_out = self.SpatialGate(x_out)
        return x_out


class ConCsNet(nn.Module):
    def __init__(self, num_class, mid_fea, fea_num):
        super().__init__()

        self.encoder = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)

        num_ftrs = self.encoder.fc.in_features
        # print(num_ftrs)
        modules = list(self.encoder.children())[:-2]
        self.encoder = nn.Sequential(*modules)
        self.cbam = CBAM(num_ftrs, pool_types=['avg'])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Linear(num_ftrs, mid_fea),
            nn.ReLU(),
            nn.Linear(mid_fea, num_class),
        )
        self.projector = nn.Sequential(
            nn.Linear(num_ftrs, mid_fea),
            nn.ReLU(),
            nn.Linear(mid_fea, fea_num),
        )

    def forward(self, x):
        out = self.encoder(x)
        # print(out.shape)
        out = self.cbam(out)
        out = torch.flatten(self.avgpool(out), start_dim=1)

        fea_classifier = self.classifier(out)
        fea_projector = self.projector(out)
        fea_projector = F.normalize(fea_projector, dim=1)
        return fea_projector, fea_classifier
        # return out


class CLSNet(nn.Module):
    def __init__(self, num_class, mid_fea, fea_num):
        super().__init__()

        # self.encoder = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        self.encoder = models.resnet50()
        num_ftrs = self.encoder.fc.in_features
        # print(num_ftrs)
        modules = list(self.encoder.children())[:-2]

        self.encoder = nn.Sequential(*modules)
        self.cbam = CBAM(num_ftrs, pool_types=['avg'])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Linear(num_ftrs, mid_fea),
            nn.ReLU(),
            nn.Linear(mid_fea, num_class),
        )

    def forward(self, x):
        print(list(self.encoder.children()))
        out = self.encoder(x)
        # print(out.shape)
        # out = self.cbam(out)
        out = torch.flatten(self.avgpool(out), start_dim=1)
        fea_classifier = self.classifier(out)
        return fea_classifier


if __name__ == '__main__':
    b = 8
    x = torch.randn((b, 3, 224, 224))
    y = torch.randn((b, 3, 224, 224))
    label = torch.randint(0, 2, [b])
    # criterion = SupConLoss()
    # print(label)
    model = ConCsNet(5, 512, 512)
    model2 = CLSNet(5, 512, 512)
    # for k,v in model.state_dict().items():
    #     print(k)
    fea1, soft1 = model(x)
    fea2 = model2(y)
    # out = torch.nn.functional.softmax(soft1, dim=1)
    # out = torch.argmax(out, dim=1)
    # print(out)
    # loss = criterion(torch.cat([fea1.unsqueeze(1), fea2.unsqueeze(1)], dim=1),labels=label)
    # print(soft1, soft2)
    # loss = criterion(soft1, label) + criterion(soft2, label)
    # print(loss.item())
    # torchsummary.summary(model.cuda(), (3, 224, 224))
