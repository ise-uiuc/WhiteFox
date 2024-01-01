
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_a = torch.nn.Conv2d(1, 1, 1, stride=1, padding=0)
    def forward(self, x1):
        v1 = self.conv_a(x1)
        return v1
# Inputs to the model
x1 = torch.randn(1, 1, 64, 64)
