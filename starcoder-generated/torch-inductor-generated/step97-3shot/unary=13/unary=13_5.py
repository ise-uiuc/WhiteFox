
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(3, 6)
        self.sigm = torch.nn.Sigmoid()
 
    def forward(self, x1):
        a1 = self.linear(x1)
        z1 = self.sigm(a1)
        o1 = a1 * z1
        return o1

# Initialzing the model
m = Model()

# Inputs to the model
__input__ = torch.randn(2, 3)
