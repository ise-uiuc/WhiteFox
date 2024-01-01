
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 16, 3, 1, 1)
        self.conv2 = torch.nn.Conv2d(16, 64, 3, 1, 1)
        self.drop = torch.nn.Dropout(0.3)
    def forward(self, x):
        v1 = self.conv1(x)
        v2 = self.conv2(v1)
        return self.drop(v2)
# Inputs to the model
x = torch.randn(1, 3, 32, 32)
