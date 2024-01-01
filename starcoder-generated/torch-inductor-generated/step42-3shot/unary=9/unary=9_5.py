
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 8, 1, stride=1, padding=1)
    def forward(self, x1):
        x1 = x1.add_(3)
        x1 = x1.clamp_(min=0, max=6)
        x1 = x1.div_(6)
        return x1
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
