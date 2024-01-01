
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(1, 3, 1)
    def forward(self, x1):
        v1 = x1.permute(0, 2, 1)
        return x1
# Inputs to the model
x1 = torch.randn(1, 2, 2)
