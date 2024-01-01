
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(1, 4, kernel_size=1, stride=1, padding=0)
    def forward(self, x):
        v1 = self.conv1(x)
        return torch.mul(v1, x)
# Inputs to the model
x = torch.randn(1, 1, 20, 20)
