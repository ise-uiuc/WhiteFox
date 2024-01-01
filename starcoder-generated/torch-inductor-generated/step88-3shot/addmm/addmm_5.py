
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        x2 = torch.randn((2,2))
        out = torch.mm(x2, x.view(9))
        return out
# Inputs to the model
x = torch.randn(1)
