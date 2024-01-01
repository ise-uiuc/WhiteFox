
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv2d = torch.nn.Conv2d(9, 9, 9, 9, bias=True)
    def forward(self, x1):
        v1 = self.conv2d(x1)
        v1 = v1.permute(0, 3, 1, 2)
        return v1
# Inputs to the model
x1 = torch.randn(1, 9, 9, 9)
