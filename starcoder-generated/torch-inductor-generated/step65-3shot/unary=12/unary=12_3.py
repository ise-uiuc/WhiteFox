
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(8,1,kernel_size=2, stride=2, padding=3, dilation=1, groups=1, bias=False)
        self.pool = nn.MaxPool2d(2,stride=2, padding=0, dilation=0, return_indices=False, ceil_mode=False)
    
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = v1.sigmoid()
        v2 = v1.mul(v2)
        v3 = self.pool(v2)
        v4 = v3.sigmoid()
        v4 = v3.mul(v4)
        return v4
# Inputs to the model
x1 = torch.randn(1, 8, 64, 64)
