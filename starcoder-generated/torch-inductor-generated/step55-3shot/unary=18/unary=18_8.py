
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(128, 64, kernel_size=1, stride=1, padding=0, bias=True)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=1, stride=1, padding=0, bias=True)
     
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = torch.sigmoid(v1)
        v3 = self.conv2(v2)
        v4 = torch.sigmoid(v3)
        return v4
# Inputs to the model
x1 = torch.randn(1, 128, 17, 17)
