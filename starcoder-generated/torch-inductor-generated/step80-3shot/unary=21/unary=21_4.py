
class ModelTanh(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(in_channels=3, out_channels=1, kernel_size=3, padding=(1, 1), stride=(2, 2))
        self.conv2 = torch.nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, padding=(2, 2), stride=(4, 4))
        self.conv3 = torch.nn.Conv2d(in_channels=1, out_channels=3, kernel_size=3, padding=(1, 1), stride=(1, 1))
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = self.conv2(v1)
        v3 = self.conv3(v2)
        v4 = torch.tanh(v3)
        v5 = torch.zeros_like(v4)
        return v5
# Inputs to the model
x1 = torch.randn(10, 3, 28, 28)
