import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    """
    A residual block that performs two 3D convolutions with Instance Normalization and LeakyReLU activations.
    Adds a skip connection from input to output to facilitate gradient flow.
    """
    def __init__(self, in_channels, out_channels, negative_slope=0.01, dropout_p=0.1):
        """
        Initializes the ResidualBlock.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            negative_slope (float): Negative slope for LeakyReLU activation.
            dropout_p (float): Dropout probability.
        """
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.InstanceNorm3d(out_channels)
        self.relu = nn.LeakyReLU(negative_slope, inplace=True)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.InstanceNorm3d(out_channels)
        self.skip = nn.Conv3d(in_channels, out_channels, kernel_size=1, padding=0)
        self.dropout = nn.Dropout3d(p=dropout_p)

    def forward(self, x):
        """
        Forward pass of the ResidualBlock.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after applying residual block.
        """
        identity = self.skip(x).type(x.dtype)
        out = self.relu(self.bn1(self.conv1(x)).type(x.dtype))
        out = self.bn2(self.conv2(out)).type(x.dtype)
        out = self.dropout(out)
        out += identity
        return self.relu(out).type(x.dtype)

class AttentionBlock(nn.Module):
    """
    An attention block that performs attention mechanism to focus on important features.
    """
    def __init__(self, F_g, F_l, F_int, negative_slope=0.01):
        """
        Initializes the AttentionBlock.

        Args:
            F_g (int): Number of channels in gating signal.
            F_l (int): Number of channels in the input feature map.
            F_int (int): Number of intermediate channels.
            negative_slope (float): Negative slope for LeakyReLU activation.
        """
        super(AttentionBlock, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv3d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.InstanceNorm3d(F_int)
        )
        self.W_x = nn.Sequential(
            nn.Conv3d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.InstanceNorm3d(F_int)
        )
        self.psi = nn.Sequential(
            nn.Conv3d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.InstanceNorm3d(1),
            nn.Sigmoid()
        )
        self.relu = nn.LeakyReLU(negative_slope, inplace=True)

    def forward(self, g, x):
        """
        Forward pass of the AttentionBlock.

        Args:
            g (torch.Tensor): Gating signal.
            x (torch.Tensor): Input feature map.

        Returns:
            torch.Tensor: Output tensor after applying attention mechanism.
        """
        g1 = self.W_g(g).type(g.dtype)
        x1 = self.W_x(x).type(x.dtype)
        psi = self.relu(g1 + x1).type(g.dtype)
        psi = self.psi(psi).type(g.dtype)
        return x * psi

class Down(nn.Module):
    """
    A downsampling block that performs max pooling followed by a residual block.
    """
    def __init__(self, in_channels, out_channels, negative_slope=0.01, dropout_p=0.1):
        """
        Initializes the Down block.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            negative_slope (float): Negative slope for LeakyReLU activation.
            dropout_p (float): Dropout probability.
        """
        super(Down, self).__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool3d(2),
            ResidualBlock(in_channels, out_channels, negative_slope, dropout_p)
        )

    def forward(self, x):
        """
        Forward pass of the Down block.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after downsampling.
        """
        return self.maxpool_conv(x)

class Up(nn.Module):
    """
    An upsampling block that performs upsampling followed by a residual block and concatenation.
    """
    def __init__(self, in_channels, out_channels, trilinear=True, negative_slope=0.01, dropout_p=0.1):
        """
        Initializes the Up block.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            trilinear (bool): If True, uses trilinear upsampling. Otherwise, uses transposed convolution.
            negative_slope (float): Negative slope for LeakyReLU activation.
            dropout_p (float): Dropout probability.
        """
        super(Up, self).__init__()
        
        if trilinear:
            self.up = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose3d(in_channels // 2, in_channels // 2, kernel_size=2, stride=2)

        self.conv = ResidualBlock(in_channels, out_channels, negative_slope, dropout_p)

    def forward(self, x1, x2):
        """
        Forward pass of the Up block.

        Args:
            x1 (torch.Tensor): Input tensor from the previous layer.
            x2 (torch.Tensor): Skip connection tensor from the encoder.

        Returns:
            torch.Tensor: Output tensor after upsampling and concatenation.
        """
        x1 = self.up(x1)
        diffZ = x2.size()[2] - x1.size()[2]
        diffY = x2.size()[3] - x1.size()[3]
        diffX = x2.size()[4] - x1.size()[4]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2,
                        diffZ // 2, diffZ - diffZ // 2])

        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class OutConv(nn.Module):
    """
    The output convolutional layer that maps the features to the desired number of output channels.
    """
    def __init__(self, in_channels, out_channels):
        """
        Initializes the OutConv block.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
        """
        super(OutConv, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        """
        Forward pass of the OutConv block.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor with the desired number of output channels.
        """
        return self.conv(x)

class UNet3D(nn.Module):
    """
    The 3D U-Net model with residual blocks and attention mechanisms.
    """
    def __init__(self, in_channels, out_channels, trilinear=True, negative_slope=0.01, dropout_p=0.1):
        """
        Initializes the UNet3D model.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            trilinear (bool): If True, uses trilinear upsampling. Otherwise, uses transposed convolution.
            negative_slope (float): Negative slope for LeakyReLU activation.
            dropout_p (float): Dropout probability.
        """
        super(UNet3D, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.trilinear = trilinear
        self.negative_slope = negative_slope
        self.dropout_p = dropout_p

        self.inc = ResidualBlock(in_channels, 64, negative_slope, dropout_p)
        self.down1 = Down(64, 128, negative_slope, dropout_p)
        self.down2 = Down(128, 256, negative_slope, dropout_p)
        self.down3 = Down(256, 384, negative_slope, dropout_p)
        self.down4 = Down(384, 512, negative_slope, dropout_p)
        self.down5 = Down(512, 768, negative_slope, dropout_p)
        self.down6 = Down(768, 1024, negative_slope, dropout_p)  # Additional down layer

        factor = 2 if trilinear else 1

        self.up1 = nn.Sequential(
            Up(1024 + 768, 768 // factor, trilinear, negative_slope, dropout_p),
            AttentionBlock(768 // factor, 768, 384, negative_slope)
        )
        self.up2 = nn.Sequential(
            Up(768 + 512, 512 // factor, trilinear, negative_slope, dropout_p),
            AttentionBlock(512 // factor, 512, 256, negative_slope)
        )
        self.up3 = nn.Sequential(
            Up(512 + 384, 384 // factor, trilinear, negative_slope, dropout_p),
            AttentionBlock(384 // factor, 384, 192, negative_slope)
        )
        self.up4 = nn.Sequential(
            Up(384 + 256, 256 // factor, trilinear, negative_slope, dropout_p),
            AttentionBlock(256 // factor, 256, 128, negative_slope)
        )
        self.up5 = nn.Sequential(
            Up(256 + 128, 128 // factor, trilinear, negative_slope, dropout_p),
            AttentionBlock(128 // factor, 128, 64, negative_slope)
        )
        self.up6 = nn.Sequential(
            Up(128 + 64, 64 // factor, trilinear, negative_slope, dropout_p),
            AttentionBlock(64 // factor, 64, 32, negative_slope)
        )
        self.outc = OutConv(64, out_channels)

    def forward(self, x):
        """
        Forward pass of the UNet3D model.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x6 = self.down5(x5)
        x7 = self.down6(x6)

        x = self.up1[0](x7, x6).type(x6.dtype)
        x = self.up1[1](x, x6).type(x6.dtype)
        x = self.up2[0](x, x5).type(x5.dtype)
        x = self.up2[1](x, x5).type(x5.dtype)
        x = self.up3[0](x, x4).type(x4.dtype)
        x = self.up3[1](x, x4).type(x4.dtype)
        x = self.up4[0](x, x3).type(x3.dtype)
        x = self.up4[1](x, x3).type(x3.dtype)
        x = self.up5[0](x, x2).type(x2.dtype)
        x = self.up5[1](x, x2).type(x2.dtype)
        x = self.up6[0](x, x1).type(x1.dtype)
        x = self.up6[1](x, x1).type(x1.dtype)

        logits = self.outc(x)
        return logits

  
