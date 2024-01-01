
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layer_one = torch.nn.Conv2d(16, 15, 1, stride=1, padding=2)
    def forward(self, x, other_conv=None, other_dense=None, x1=None):
        if other_conv == None:
            other_conv = torch.randn(1, 10, 16, 16)
        if x1 == None:
            x1 = torch.randn(1, 1, 256, 256)
        v1 = self.layer_one(x)
        var2 = v1 + other_conv
        if other_dense == None:
            other_dense = torch.randn(1, 10)
        var3 = var2 + other_dense
        return var3
# Inputs to the model
x = torch.randn(1, 16, 256, 256)
x1 = torch.randn(1, 1, 256, 256)
