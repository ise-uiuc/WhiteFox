
class ConvBlock(nn.Module):
    def __init__(self):
        super(ConvBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=8, kernel_size=3, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(8, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn(x)
        return x    

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.convblock =  ConvBlock() 
        self.batch_norm = nn.BatchNorm2d(8, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.conv = nn.Conv2d(in_channels=8, out_channels=8, kernel_size=3, padding=1, bias=False)
    @torch.no_grad()
    def forward(self, x):
        x = self.convblock(x)
        x = self.batch_norm(x)
        x = self.conv(x)
        return x
#Inputs to the model
x = torch.randn(1, 1, 4, 4)
