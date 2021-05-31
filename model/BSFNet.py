import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import warnings
warnings.filterwarnings(action='ignore')
'''
BSFNet: bilateral semantic fusion siamese network for change detection
shallow structure feature extractor: conv downsample 4x, spatial attention
deep semantic feature extractor: resnet50 downsample 32x, channel attention
BAM: deep context features reverse activation to refine shallow spatial features
FFM = ARM(16x down) + ARM(32x down) without residual structure
'''
class resnet50(nn.Module):
    def __init__(self, pretrained=True):
        super(resnet50, self).__init__()
        self.features = models.resnet50(pretrained=pretrained)
        self.conv1 = self.features.conv1
        self.bn1 = self.features.bn1
        self.relu = self.features.relu
        self.maxpool1 = self.features.maxpool
        self.layer1 = self.features.layer1
        self.layer2 = self.features.layer2
        self.layer3 = self.features.layer3
        self.layer4 = self.features.layer4
    def forward(self, input):
        x = self.conv1(input)
        x = self.relu(self.bn1(x))
        x = self.maxpool1(x)
        feature1 = self.layer1(x) # 1/4
        feature2 = self.layer2(feature1) # 1/8
        feature3 = self.layer3(feature2) # 1/16
        feature4 = self.layer4(feature3) # 1/32

        # global average pooling to build tail
        tail = torch.mean(feature4, 3, keepdim=True)
        tail = torch.mean(tail, 2, keepdim=True)
        return feature3, feature4, tail

class convblock(nn.Module):
    def __init__(self, inchannels, outchannels, ks=3, stride=2, padding=1):
        super(convblock, self).__init__()
        self.conv = nn.Conv2d(inchannels,outchannels, kernel_size=ks, stride=stride, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(outchannels)
        self.relu = nn.ReLU(inplace=True)
    def forward(self, input):
        x = self.conv(input)
        return self.relu(self.bn(x))

class spatial_path(nn.Module):
    def __init__(self):
        super(spatial_path, self).__init__()
        self.conv1 = convblock(3, 64, ks=7, stride=2, padding=3)
        self.conv2 = convblock(64, 128,  ks=7, stride=2, padding=3)
    def forward(self, input):
        layer1 = self.conv1(input)
        layer2 = self.conv2(layer1)
        return layer1, layer2

class PAM(nn.Module):
    # spatial attention for low-level feature layers
    def __init__(self, inchannels, ksize=9):
        super(PAM, self).__init__()
        self.inchannels = inchannels
        inter_channels = inchannels // 2
        self.conv_l1 = nn.Conv2d(inchannels, inter_channels, kernel_size=(1, ksize), padding=(0, int((ksize-1)/2)))
        self.bn_l1 = nn.BatchNorm2d(inter_channels)
        self.conv_l2 = nn.Conv2d(inter_channels, 1, kernel_size=(ksize, 1), padding=(int((ksize - 1) / 2), 0))
        self.bn_l2 = nn.BatchNorm2d(1)
        self.conv_r1 = nn.Conv2d(inchannels, inter_channels, kernel_size=(ksize, 1), padding=(int((ksize-1)/2), 0))
        self.bn_r1 = nn.BatchNorm2d(inter_channels)
        self.conv_r2 = nn.Conv2d(inter_channels, 1, kernel_size=(1, ksize), padding=(0, int((ksize-1)/2)))
        self.bn_r2 = nn.BatchNorm2d(1)
        self.sigmoid = nn.Sigmoid()
    def forward(self, input):
        x_l = self.bn_l1(self.conv_l1(input))
        x_l = self.bn_l2(self.conv_l2(x_l))
        x_r = self.bn_r1(self.conv_r1(input))
        x_r = self.bn_r2(self.conv_r2(x_r))
        x = x_l + x_r
        x = self.sigmoid(x)
        x = torch.mul(x, input)
        return x


class CAM(nn.Module):
    # channel attention for high-level feature layers
    def __init__(self, inchannels, outchannels):
        super(CAM, self).__init__()
        self.inchannels = inchannels
        self.conv = nn.Conv2d(inchannels, outchannels, kernel_size=1, stride=1)
        self.bn = nn.BatchNorm2d(outchannels)
        self.sigmoid = nn.Sigmoid()
        self.pool = nn.AdaptiveAvgPool2d(output_size=(1,1))
    def forward(self, input):
        x = self.pool(input)
        assert self.inchannels == x.size(1)
        x = self.conv(x)
        x = self.sigmoid(self.bn(x))
        # x = self.sigmoid(x)
        x = torch.mul(x, input)
        return x

class BAM(nn.Module):
    def __init__(self, inchannels, channels):
        super(BAM, self).__init__()
        self.conv1x1_1 = nn.Conv2d(inchannels, channels, kernel_size=1)
        self.activation = nn.Sigmoid()
        self.conv1x1_2 = nn.Conv2d(channels, channels, kernel_size=1)
        self.conv3x3_1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
    def forward(self, shallow_layer, deep_layer):
        dout = self.conv1x1_1(deep_layer)
        dout = 1 - self.activation(dout)
        fuse_out = torch.mul(shallow_layer, dout)
        b_out = self.conv3x3_1(self.conv1x1_2(fuse_out))
        return b_out

class FFM(nn.Module):
    '''feature fusion module'''
    def __init__(self, num_classes, inchannels):
        super(FFM, self).__init__()
        self.inchannels = inchannels
        self.pool = nn.AdaptiveAvgPool2d(output_size=(1,1))
        self.conv1 = nn.Conv2d(64, 64, kernel_size=1)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(64, 64, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
        self.convblock = convblock(inchannels, 64, stride=1)
        self.final_conv = convblock(64, num_classes, stride=1)
    def forward(self, input1, input2):
        x = torch.cat((input1, input2), dim=1)
        assert self.inchannels == x.size(1)
        feature = self.convblock(x)
        x = self.pool(feature)
        x = self.relu(self.conv1(x))
        x = self.sigmoid(self.conv2(x))
        output1 = torch.mul(x, feature)
        output1 = self.final_conv(output1)
        # output2 = torch.add(feature, output1)
        return output1

class BSFNet(nn.Module):
    def __init__(self, num_classes, train=False):
        super(BSFNet, self).__init__()
        self.training = train
        self.spatial_path = spatial_path()
        self.context_path = resnet50(pretrained=True)

        self.conv3x3_sx = nn.Conv2d(192, 64, kernel_size=3, padding=1)
        self.pam1 = PAM(64, ksize=7)
        self.cam1 = CAM(1024, 1024)
        self.cam2 = CAM(2048, 2048)
        self.conv1x1_cx1 = nn.Conv2d(1024, 256, kernel_size=1)
        self.conv1x1_cx2 = nn.Conv2d(2048, 256, kernel_size=1)
        self.bam = BAM(256, 64)
        self.supervision1 = nn.Conv2d(256, num_classes, kernel_size=1)
        self.supervision2 = nn.Conv2d(256, num_classes, kernel_size=1)
        self.ffm = FFM(num_classes, 256+64)

        self.conv = nn.Conv2d(num_classes, num_classes, kernel_size=1)
        self.init_weights()

    def init_weights(self):
        for name, m in self.named_modules():
            if 'context_path' not in name:
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                elif isinstance(m, nn.BatchNorm2d):
                    m.eps = 1e-5
                    m.momentum = 0.1
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)

    def forward(self, input):
        layer1, layer2 = self.spatial_path(input)
        layer2_up = F.upsample(layer2, size=layer1.size()[2:], mode='bilinear', align_corners=True)
        sx_cat = torch.cat((layer1, layer2_up), dim=1)
        sx = self.conv3x3_sx(sx_cat)
        sx = self.pam1(sx)

        cx1, cx2, tail = self.context_path(input)
        cx1 = self.cam1(cx1)
        cx2 = self.cam2(cx2)
        cx2 = cx2 + tail
        cx1 = self.conv1x1_cx1(cx1)
        cx2 = self.conv1x1_cx2(cx2)
        # context path upsample
        cx1 = F.interpolate(cx1, size=sx.size()[-2:], mode='bilinear')
        cx2 = F.interpolate(cx2, size=sx.size()[-2:], mode='bilinear')
        cx = cx1 + cx2

        bout = self.bam(sx, cx)

        if self.training:
            cx1_sup = self.supervision1(cx1)
            cx2_sup = self.supervision2(cx2)
            cx1_sup = F.interpolate(cx1_sup, size=input.size()[-2:], mode='bilinear')
            cx2_sup = F.interpolate(cx2_sup, size=input.size()[-2:], mode='bilinear')

        result = self.ffm(bout, cx)
        # upsample
        result = F.interpolate(result, scale_factor=2, mode='bilinear')
        result = self.conv(result)

        # if self.training:
        #     return result, cx1_sup, cx2_sup
        return result

class SiameseNet(nn.Module):
    def __init__(self, norm_flag='l2'):
        super(SiameseNet, self).__init__()
        self.basenet = BSFNet(num_classes=16)
        if norm_flag == 'l2':
            self.norm = F.normalize
        elif norm_flag == 'exp':
            self.norm = nn.Softmax2d()
    def forward(self, t0, t1):
        out_t0 = self.basenet(t0)
        out_t1 = self.basenet(t1)
        out_t0_conv_norm, out_t1_conv_norm = self.norm(out_t0, 2, dim=1), self.norm(out_t1, 2, dim=1)
        return [out_t0_conv_norm, out_t1_conv_norm]

if __name__ == '__main__':
    model = SiameseNet()
    x1 = torch.rand(4, 3, 256, 256)
    x2 = torch.rand(4, 3, 256, 256)
    output = model(x1, x2)
    print(output[0].size())






