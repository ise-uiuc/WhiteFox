
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(8, 16)
 
    def forward(self, x1):
        c = torch.rand(size=(5,))
        v1 = self.linear(x1)
        v2 = v1 - c
        return v2

# Initializing the model
m = Model()

# Inputs to the model
__input__ = torch.randn(5, 8)
