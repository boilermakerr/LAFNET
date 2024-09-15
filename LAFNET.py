import torch
import torch.nn as nn
import math




def autopad(k, p=None, d=1):
    """
    `k`: kernel, `p`: padding, `d`: dilation.
    """
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p

class Conv(nn.Module):

    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        return self.act(self.conv(x))

class DWConv(Conv):
    # Depth-wise convolution
    def __init__(self, c1, c2, k=1, s=1, d=1, act=True):

        super().__init__(c1, c2, k, s, g=math.gcd(c1, c2), d=d, act=act)


class GhostConv(nn.Module):
    # Ghost Convolution
    def __init__(self, c1, c2, k=1, s=1, g=1, act=True):
        super().__init__()
        c_ = c2 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, k, s, None, g, act=act)
        self.cv2 = Conv(c_, c_, 5, 1, None, c_, act=act)

    def forward(self, x):
        y = self.cv1(x)
        return torch.cat((y, self.cv2(y)), 1)





class LightConv(nn.Module):

    def __init__(self, c1, c2, k=1, act=nn.ReLU()):
        super().__init__()
        self.conv1 = Conv(c1, c2, 1, act=False)
        self.conv2 = DWConv(c2, c2, k, act=act)

    def forward(self, x):
        return self.conv2(self.conv1(x))


class LCGBlock(nn.Module):
    """LG Block"""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """Initialize 'SPP' module with various pooling sizes for spatial pyramid pooling."""
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        self.block1 = LightConv(c1, c_, 1, act=False)
        self.block2 = LightConv(c1, c_, 1, act=False)
        self.block3 = Conv(2 * c_, c2, 1)
        self.m = nn.Sequential(*(LGhostBottleneck(c_, c_) for _ in range(n)))

    def forward(self, x):
        return self.block3(torch.cat((self.m(self.block1(x)), self.block2(x)), 1))

class LGhostBottleneck(nn.Module):
    """Ghost Block"""

    def __init__(self, c1, c2, k=3, s=1):  # ch_in, ch_out, kernel, stride
        super().__init__()
        c_ = c2 // 2
        self.conv = nn.Sequential(
            GhostConv(c1, c_, 1, 1),  # pw
            DWConv(c_, c_, k, s, act=False) if s == 2 else nn.Identity(),  # dw
            GhostConv(c_, c2, 1, 1, act=False))  # pw-linear
        self.shortcut = nn.Sequential(DWConv(c1, c1, k, s, act=False), Conv(c1, c2, 1, 1,act=False)) if s == 2 else nn.Identity()
        self.shortcut2= nn.Sequential(LightConv(c1, c2, 1, act=False))
        self.Pointwise_Convolution1 = nn.Conv2d(in_channels=c1,
                                    out_channels=c2,
                                    kernel_size=1,
                                    stride=1,
                                    padding=0,
                                    groups=1)
        self.Pointwise_Convolution2 = nn.Conv2d(in_channels=c1,
                                    out_channels=c2,
                                    kernel_size=1,
                                    stride=1,
                                    padding=0,
                                    groups=1)
    def forward(self, x):
        return self.conv(x)+self.shortcut(x)
