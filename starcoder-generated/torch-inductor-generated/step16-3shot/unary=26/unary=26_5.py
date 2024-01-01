
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_t = torch.nn.ConvTranspose2d(256, 256, 5, stride=2, padding=1)
        self.conv_t1 = torch.nn.ConvTranspose2d(256, 256, 4, stride=3, padding=0)
        self.conv_t3 = torch.nn.ConvTranspose2d(256, 256, 7, stride=2, padding=1)
        self.conv_c5 = torch.nn.Conv2d(256, 256, 5, stride=1, padding=2)
        self.conv_t7 = torch.nn.ConvTranspose2d(256, 128, 2, stride=1, padding=0)
        self.conv_t8 = torch.nn.ConvTranspose2d(128, 64, 3, stride=1, padding=0)
        self.conv_t10 = torch.nn.ConvTranspose2d(64, 2, 2, stride=2, padding=0)
    def forward(self, x9):
        x1, x2, x3, x4, x5, x6, x7 = x9.split([64, 160, 160, 16, 320, 64, 128], 1)
        x8 = x7 > 0  
        x9 = x7 * -0.15   
        x10 = torch.where(x8, x7, x9)
        return torch.cat([x1, x2, x3, x4, x5, x6, x10], 1)
# Inputs to the model
x9 = torch.randn(8, 512, 4, 4)
