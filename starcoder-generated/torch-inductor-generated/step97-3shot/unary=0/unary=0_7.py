
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv17 = torch.nn.Conv2d(8, 9, 2, stride=1, padding=3)
    def forward(self, x):
        v17 = self.conv17(x)
        return v17
# Inputs to the model
x = torch.randn(1, 8, 8, 8)
