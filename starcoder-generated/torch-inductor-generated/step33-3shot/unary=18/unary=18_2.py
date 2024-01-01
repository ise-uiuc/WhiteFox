
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.conv2 = torch.nn.Conv2d(in_channels=16, out_channels=1, kernel_size=5, stride=1, padding=1)
    def forward(self, x1):
        v1 = nn.Conv2d(1, 16, 3, 1, 1)(x1)
        v2 = nn.Sigmoid()(v1)
        v3 = nn.Conv2d(16, 1, 5, 1, 1)(v2)
        v4 = nn.Sigmoid()(v3)
        return v4
# Inputs to the model
x1 = torch.randn(1, 1, 225, 225)
