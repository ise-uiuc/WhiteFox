
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 16, kernel_size=3, bias=False)
    def forward(self, x):
        v1 = self.conv(x)
        v2 = torch.tanh(v1)
        v3 = torch.tanh(v1)
        return torch.mm(v3.flatten(1), v2.flatten(1).T)
# Inputs to the model
x = torch.randn(1, 3, 224, 244)
