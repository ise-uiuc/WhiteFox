
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 16, kernel_size=(1, 1), stride=(1, 1))
        self.conv2 = torch.nn.Conv2d(16, 16, kernel_size=(1, 1), stride=(1, 1))
        self.conv3 = torch.nn.Conv2d(16, 16, kernel_size=(3, 1), stride=(1, 2), padding=(1, 0))
    def forward(self, x):
        v1 = self.conv1(x)
        v2 = self.conv2(v1)
        v3 = self.conv3(v2)
        v4 = v3 - 26.3
        return v4
# Inputs to the model
x = torch.randn(1, 3, 8, 32)
