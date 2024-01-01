
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv2 = torch.nn.Conv2d(64, 64, 3, stride=2, padding=1)
    def forward(self, x1):
        return self.conv2(F.relu(x1 - 1.5))
# Inputs to the model
x1 = torch.randn(1, 64, 16, 16)
