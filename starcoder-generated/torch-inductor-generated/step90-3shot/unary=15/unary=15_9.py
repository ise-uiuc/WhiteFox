
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 7, 2, 2, 1)
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = torch.nn.functional.tanh(v1)
        return v1
# Inputs to the model
inputs = torch.randn(1,3,256,256)
