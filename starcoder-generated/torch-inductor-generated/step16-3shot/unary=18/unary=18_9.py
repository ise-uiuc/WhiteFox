
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(in_channels=3, out_channels=220, kernel_size=1, stride=1, padding=1)
    def forward(self, x1):
        v1 = torch.sigmoid(self.conv1(x1))
        v2 = torch.relu(v1)
        return v2
# Inputs to the model
x1 = torch.randn(3, 3, 259, 259)
