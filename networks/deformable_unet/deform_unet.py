from .deform_part import deform_up, deform_down, deform_inconv
from .unet_parts import *



class DUNetV1V2(nn.Module):
    # downsize_nb_filters_factor=4 compare to DUNetV1V2_MM
    def __init__(self, n_channels, n_classes, downsize_nb_filters_factor=4):
        super(DUNetV1V2, self).__init__()
        # self.inc = deform_inconv(n_channels, 64 // downsize_nb_filters_factor)
        self.inc = inconv(n_channels, 64 // downsize_nb_filters_factor)
        self.down1 = deform_down(64 // downsize_nb_filters_factor, 128 // downsize_nb_filters_factor)
        self.down2 = deform_down(128 // downsize_nb_filters_factor, 256 // downsize_nb_filters_factor)
        self.down3 = down(256 // downsize_nb_filters_factor, 512 // downsize_nb_filters_factor)
        self.down4 = down(512 // downsize_nb_filters_factor, 512 // downsize_nb_filters_factor)
        self.up1 = up(1024 // downsize_nb_filters_factor, 256 // downsize_nb_filters_factor)
        self.up2 = up(512 // downsize_nb_filters_factor, 128 // downsize_nb_filters_factor)
        self.up3 = deform_up(256 // downsize_nb_filters_factor, 64 // downsize_nb_filters_factor)
        self.up4 = deform_up(128 // downsize_nb_filters_factor, 64 // downsize_nb_filters_factor)
        self.outc = nn.Conv2d(64 // downsize_nb_filters_factor + n_channels, n_classes, 1)

    def forward(self, inp):
        x1 = self.inc(inp)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = torch.cat([inp, x], dim=1)
        x = self.outc(x)
        # return torch.sigmoid(x)
        return x
