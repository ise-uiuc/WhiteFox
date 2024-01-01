
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 8, 1, stride=1, padding=1)
    def forward(self, x1):
        v2 = torch.div(torch.clamp(self.conv(x1) + 3, min=0, max=6), 6)
        return v2
    def eval(self, x1):
        # For example, if you are to enable training on an exported model:
        return torch.div(torch.clamp(self.conv(x1) + 3, min=0, max=6), 6)
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
