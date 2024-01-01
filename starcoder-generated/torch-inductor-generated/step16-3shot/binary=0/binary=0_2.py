
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(1, 3, 1, stride=1, padding=1)
    def forward(self, x1, x2, argmax=None):
        v1 = torch.flatten(x1, 1)
        v2 = self.conv(v1)
        if argmax == None:
            argmax = torch.argmax(v2, dim=0, keepdim=True)
        return x2 + argmax
# Inputs to the model
x1 = torch.randn(1, 1, 3, 64, 64)
x2 = torch.randn(3, 64, 64)
