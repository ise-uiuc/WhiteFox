
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, param0):
        var2 = torch.Tensor.permute(param0, 1, 0)
        var1 = var2 - param0
        var0 = torch.bmm(param0, var1)
        return var0
# Inputs to the model
param0 = torch.randn(2, 2, 2)
