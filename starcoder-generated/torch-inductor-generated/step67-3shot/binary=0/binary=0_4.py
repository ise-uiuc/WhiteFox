
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(1, 1, 3, stride=3, padding=0)
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = v1 + 1
        return v2
# Inputs to the model
x1 = torch.Tensor(1, 1, 2, 2)
