
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(32, 16, kernel_size=(1, 1), stride=(1, 1), bias=False, padding=(0, 0))
        self.conv2 = nn.Conv2d(16, 64, kernel_size=(2, 1), stride=(1, 1), bias=False)
    def forward(self, x1):
        v1 = self.conv1(x1)
        return v1 + 3
# Inputs to the model
x1 = torch.randn(1, 32, 32, 32)
