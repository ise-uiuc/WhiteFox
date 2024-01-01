
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 4, 1, 3, padding=2)
    def forward(self, x):
        return torch.sigmoid(self.conv(x)) * 3 + torch.ones(1, 1, 4, 4) ** 2
# Inputs to the model
x = torch.randn(1,3,32,32)
