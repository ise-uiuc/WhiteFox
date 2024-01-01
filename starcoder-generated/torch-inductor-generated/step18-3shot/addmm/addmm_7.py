
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1, x2, inp):
        v1 = torch.mm(x1, x2)
        v2 = v1.reshape(6, -1, 3)
        v3 = v2 * inp  ## Comment this if you want the model to have the specified pattern
        v4 = v3.permute(1, 0, 2)
        v5 = v4 + v2
        v6 = v5.permute(1, 0, 2)
        return v6
# Inputs to the model
x1 = torch.randn(5, 4)
x2 = torch.randn(5, 4)
inp = torch.randn(4, 3)
