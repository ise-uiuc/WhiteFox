
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.ConvTranspose2d(3, 64, 3, stride=2, padding=0, output_padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.prelu1 = nn.PReLU([64])
        self.conv2 = nn.ConvTranspose2d(64, 64, 3, stride=2, padding=0, output_padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(64)
        self.prelu2 = nn.PReLU([64])
        self.conv3 = nn.ConvTranspose2d(64, 3, 3, stride=2, padding=0, output_padding=1, bias=False)
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.prelu1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.prelu2(x)
        x = self.conv3(x)
        
# Inputs to the model
x1 = torch.randn(1, 3, 32, 32)
