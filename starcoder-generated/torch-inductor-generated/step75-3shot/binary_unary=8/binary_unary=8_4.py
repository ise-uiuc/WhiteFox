
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(in_channels=2, out_channels=8, kernel_size=(1, 3), stride=1, padding=0, bias=False)
        self.conv2 = torch.nn.Conv2d(in_channels=2, out_channels=8, kernel_size=(3, 1), stride=1, padding=0, bias=False)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = self.conv2(x1)
        v3 = v1
        v4 = v3 + v2
        v5 = torch.relu(v4)
        return v5
# Inputs to the model
x1 = torch.randn(1, 2, 32, 32)
