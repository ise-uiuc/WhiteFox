
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(2, 2, 3)
    def forward(self, x1):
        x2 = x1[:,:,0,:]
        y = self.conv(x2)
        v1 = torch.nn.functional.linear(y, self.conv.weight, self.conv.bias)
        v2 = v1.permute(3, 1, 0, 2)
        return v1
# Inputs to the model
x1 = torch.randn(1, 2, 5, 5)
