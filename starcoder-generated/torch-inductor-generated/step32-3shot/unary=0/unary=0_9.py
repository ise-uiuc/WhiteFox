
class Model(torch.nn.Module):
    def __init__(self):
        self.avgpool2d = torch.nn.AvgPool2d(kernel_size=228)
    def forward(self, x3, x2):
        x3 = self.avgpool2d(x3)
        v1 = x2 + x3
        return v1
# Inputs to the model
x3 = torch.randn(1, 3, 9, 7)
x2 = torch.randn(1, 2, 15, 13)
