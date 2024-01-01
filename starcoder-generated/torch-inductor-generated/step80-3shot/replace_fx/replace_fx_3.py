

# Model with a submodule
class Submodule(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        return x + 1

class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.submodule = Submodule()
    def forward(self, x):
        x = F.dropout(x, p=0.5)
        x = torch.rand_like(x)
        submodule_out = self.submodule.forward(x)
        return x
# Inputs to the model
x = torch.randn(1, 2, 2)
