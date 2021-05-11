from .unet_parts import *


class UNet(nn.Module):
    def __init__(self, n_channels, n_classes,downsize_nb_filters_factor=2):
        super(UNet, self).__init__()
        self.inc = inconv(n_channels, 64//downsize_nb_filters_factor)
        self.down1 = down(64//downsize_nb_filters_factor, 128//downsize_nb_filters_factor)
        self.down2 = down(128//downsize_nb_filters_factor, 256//downsize_nb_filters_factor)
        self.down3 = down(256//downsize_nb_filters_factor, 512//downsize_nb_filters_factor)
        self.down4 = down(512//downsize_nb_filters_factor, 512//downsize_nb_filters_factor)
        self.up1 = up(1024//downsize_nb_filters_factor, 256//downsize_nb_filters_factor)
        self.up2 = up(512//downsize_nb_filters_factor, 128//downsize_nb_filters_factor)
        self.up3 = up(256//downsize_nb_filters_factor, 64//downsize_nb_filters_factor)
        self.up4 = up(128//downsize_nb_filters_factor, 64//downsize_nb_filters_factor)
        self.outc = outconv(64//downsize_nb_filters_factor, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.outc(x)
        return torch.sigmoid(x)
