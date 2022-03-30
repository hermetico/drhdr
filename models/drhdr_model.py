import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import DeformConv2d



def activation(act_type="relu"):
    if act_type == "leaky":
        return nn.LeakyReLU()
    elif act_type == "relu":
        return nn.ReLU()
    elif act_type == "mish":
        return nn.Mish()
    else:
        raise ValueError(f"Unknown activation [{act_type}]")

class SpatialAttentionLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, act_type="leaky"):
        super(SpatialAttentionLayer, self).__init__()
        padding = kernel_size // 2
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, padding=padding, padding_mode="reflect", bias=True)
        self.conv2 = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, padding_mode="reflect", bias=True)
        self.act = activation(act_type)
        self.att = nn.Sigmoid()

    def forward(self, x, rx):
        xrx = torch.cat((x, rx), 1)
        xatt = self.att(self.conv2(self.act(self.conv1(xrx))))
        return xatt

class DilatedBlock(nn.Module):
    def __init__(self, in_channels, grow_rate, act_type="leaky", kernel_size=3):
        super(DilatedBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, grow_rate, kernel_size=kernel_size, padding=(kernel_size - 1) // 2 + 1, bias=True, dilation=2)
        self.act = activation(act_type)

    def forward(self, x):
        out = self.act(self.conv(x))
        out = torch.cat((x, out), 1)
        return out


class DRDB(nn.Module):
    def __init__(self, in_channels, dense_layers, grow_rate, act_type="leaky"):
        super(DRDB, self).__init__()
        num_channels = in_channels
        modules = []
        for i in range(dense_layers):
            modules.append(DilatedBlock(num_channels, grow_rate, act_type))
            num_channels += grow_rate
        self.dense_layers = nn.Sequential(*modules)
        self.conv_1x1 = nn.Conv2d(num_channels, in_channels, kernel_size=1, padding=0, bias=True)

    def forward(self, x):
        out = self.dense_layers(x)
        out = self.conv_1x1(out)
        out = out + x
        return out


class GDConv2d(nn.Module):
    """
    Guided Deformable Convolution 
    """
    def __init__(self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True,
        deformable_groups: int = 1
    ):
        super(GDConv2d, self).__init__()

        offset_channels = deformable_groups * 3 * kernel_size * kernel_size
        self.dconv = DeformConv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        self.offsetconv = nn.Conv2d(in_channels, offset_channels, kernel_size, stride, padding, dilation, groups, bias)
        #self.init_offset()
        self.mask_act = nn.Sigmoid()

        
    def init_offset(self):
        self.offsetconv.weight.data.zero_()
        self.offsetconv.bias.data.zero_()

    def forward(self, input:torch.Tensor, feat:torch.Tensor):
        """
        input: input features for deformable conv
        feat: guiding features for generating offsets and mask
        """

        out = self.offsetconv(feat)
        o1, o2, mask = torch.chunk(out, 3, dim=1)
        offset = torch.cat((o1, o2), dim=1)

        offset_mean = torch.mean(torch.abs(offset))
        if offset_mean > 100:
            print('Offset mean is {}, larger than 100.'.format(offset_mean))

        mask = self.mask_act(mask)
        return self.dconv(input, offset, mask)

class DeformableConvolutionalBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, act_type="leaky", deformable_groups=8):
        super(DeformableConvolutionalBlock, self).__init__()
        ic = in_channels
        oc = out_channels
        ks = kernel_size
        padding = ks // 2

        self.feat = nn.Sequential(
            nn.Conv2d(ic,  oc, kernel_size=3, padding=1, padding_mode="reflect", bias=True),
            activation(act_type),
            nn.Conv2d(oc,  oc, kernel_size=3, padding=1, padding_mode="reflect", bias=True),
            activation(act_type)
        )
        self.dconv = GDConv2d(oc, oc, kernel_size=ks, padding=padding, bias=True, deformable_groups=deformable_groups)
        self.act = activation(act_type)

    def forward(self, x, rx):
        """
        x (Tensor): features
        rx (Tensor): reference features
        """
        offset = self.feat(torch.cat([x, rx], dim=1))
        feat = self.act(self.dconv(x, offset))

        return feat


class DecoderBlock(nn.Module):
    def __init__(self, base_channels, kernel_size=3, act_type="leaky"):
        super(DecoderBlock, self).__init__()
        
        bc = base_channels
        ks = kernel_size
        padding = ks // 2

        self.l0 = nn.Sequential(
            nn.Conv2d(bc * 2, bc, kernel_size=ks, padding=padding, padding_mode="reflect", bias=True),
            activation(act_type)
            )

    def forward(self, l0, l1):

        o = F.interpolate(l1, l0.shape[-2:], mode="bilinear", align_corners=False)
        o = self.l0(torch.cat([o, l0], dim=1))

        return o


class SpattialAttentionBlock(nn.Module):
    """2 Levels deformable convolution"""
    def __init__(self, in_channels, out_channels, kernel_size=3, act_type="leaky"):
        super(SpattialAttentionBlock, self).__init__()
        ic = in_channels
        oc = out_channels
        ks = kernel_size

        self.attfeat = SpatialAttentionLayer(ic, oc, ks, act_type)

    def forward(self, x, rx):
        """
        x (Tensor): features
        rx (Tensor): reference features
        """
        f = self.attfeat(x, rx)
        af = x * f

        return af


class UDRDB(nn.Module):
    def __init__(self, in_channels, base_channels, dense_layers, grow_rate, act_type="leaky"):
        super(UDRDB, self).__init__()
        # 3 x DRDBs
        self.proj0 = nn.Conv2d(in_channels, base_channels, kernel_size=1, padding=0, bias=True)
        self.RDB1 = DRDB(base_channels, dense_layers, grow_rate, act_type)
        self.RDB2 = DRDB(base_channels, dense_layers, grow_rate, act_type)
        self.RDB3 = DRDB(base_channels, dense_layers, grow_rate, act_type)
        self.proj1 = nn.Conv2d(base_channels * 3, base_channels, kernel_size=1, padding=0, bias=True)
    
    def forward(self, x):
        px = self.proj0(x)
        x0 = self.RDB1(px)
        x1 = self.RDB2(x0)
        x2 = self.RDB3(x1)
        out = self.proj1(torch.cat((x0, x1, x2), 1))
        return out


class DRHDR(nn.Module):
    def __init__(self, opt=None):
        super(DRHDR, self).__init__()
        self.opt = opt

        ic = self.opt.input_channels
        bc = self.opt.base_channels
        eks = self.opt.eks
        dks = self.opt.dks
        act_type = self.opt.activation
        gr = self.opt.growth_rate
        dfg = self.opt.deformable_groups

        self.proj0 = nn.Sequential(
            nn.Conv2d(ic,  bc, kernel_size=3, padding=1, padding_mode="reflect", bias=True), 
            activation(act_type)
        )

        self.proj1 = nn.Sequential(
            nn.Conv2d(bc, bc, kernel_size=3, stride=2, padding=1, padding_mode="reflect", bias=True),
            activation(act_type)
        )

        self.attl = SpattialAttentionBlock(bc * 2, bc, eks, act_type)
        self.atth = SpattialAttentionBlock(bc * 2, bc, eks, act_type)
        
        self.defl = DeformableConvolutionalBlock(bc * 2, bc, eks, act_type,deformable_groups=dfg)
        self.defh = DeformableConvolutionalBlock(bc * 2, bc, eks, act_type,deformable_groups=dfg)

        self.ul0 = UDRDB(bc * 3, bc, 5, gr, act_type) # Unet level 0
        self.ul1 = UDRDB(bc * 3, bc, 5, gr, act_type)# Unet level 1

        self.dec = DecoderBlock(bc, dks, act_type)


        # out layer 
        chan = bc if self.opt.last_residual else bc * 2
        self.conv_out = nn.Sequential(
            nn.Conv2d(chan, bc,  kernel_size=dks, padding=1, padding_mode="reflect", bias=True),
            activation(act_type),
            nn.Conv2d(bc, 3,  kernel_size=1, padding=0, bias=True),
            nn.ReLU()
        )
    


    def forward(self, x1, x2 = None, x3 = None):
        if x2 is None:
            x2 = x1[:,1,:,:,:]
            x3 = x1[:,2,:,:,:]
            x1 = x1[:,0,:,:,:]


        # Project G
        f0l = self.proj0(x1)
        f0m = self.proj0(x2)
        f0h = self.proj0(x3)
        
        # Project  and downscale LDR
        f1l = self.proj1(f0l)
        f1m = self.proj1(f0m)
        f1h = self.proj1(f0h)



        # Def convolution
        f0l = self.defl(f0l, f0m)
        f0h = self.defh(f0h, f0m)

        # Guided Attention
        f1l = self.attl(f1l, f1m)
        f1h = self.atth(f1h, f1m)
 
        l0 = self.ul0(torch.cat([f0l, f0m, f0h], dim=1))
        l1 = self.ul1(torch.cat([f1l, f1m, f1h], dim=1))
        
        out = self.dec(l0, l1)
        if self.opt.last_residual:
            out += f0m
        else:
            out = torch.cat((out, f0m), dim=1)
        
        out = self.conv_out(out)

        return out


