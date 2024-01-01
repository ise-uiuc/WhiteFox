
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2908, 384)
 
    def forward(self, x1, x2):
        v1 = self.linear(x1)
        v3 = v1 + x2
        return v3, v1

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 2908)
x2 = torch.randn(1, 384)
__, __output2__ = m(x1, x2)

