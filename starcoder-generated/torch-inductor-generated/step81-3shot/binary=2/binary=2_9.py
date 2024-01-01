
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 3, 3, stride=2, padding=1)
        self.conv.requires_grad = True
    def forward(self, x2):
        v1 = self.conv(x2)
        v2 = v1 - 1
        return v2
# Inputs to the model
x2 = torch.randn(1, 3, 115, 115, requires_grad=True)
