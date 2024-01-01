
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(1, 8, kernel_size=(16, 16), stride=(1, 1), padding=(8, 8), dilation=(1, 1))
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = self.conv1(x1)
        v3 = self.conv1(x1)
        v4 = v1 + v2 + v3
        v5 = torch.relu(v4)
        v6 = v5.permute(0, 1, 3, 2)
        return v6
# Inputs to the model
x1 = torch.randn(1, 1, 64, 64)
