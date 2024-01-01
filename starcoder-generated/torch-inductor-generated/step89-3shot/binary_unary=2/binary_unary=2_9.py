
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(8, 128, 4)
        self.conv2 = torch.nn.Conv2d(128, 64, 1)
    def forward(self, x1):
        x2 = self.conv1(x1)
        x3 = x2 - 199
        x4 = self.conv2(x3)
        x5 = x4 - 0.555556
        x6 = F.relu(x5)
        x7 = x6 - 1
        x8 = torch.squeeze(x7, 0)
        return x8
# Inputs to the model
x1 = torch.randn(1, 8, 196, 196)
