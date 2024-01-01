
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(1, 1, 50, stride=1, padding=25)
    def forward(self, x1):
        v1 = self.conv(x1)
        return v1[0, :2, 1:49, 1:49]
# Inputs to the model
x1 = torch.randn(1, 1, 800, 800, requires_grad=True)
