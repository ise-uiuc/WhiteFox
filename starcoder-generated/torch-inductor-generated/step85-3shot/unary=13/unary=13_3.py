
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(5, 10)
 
    def forward(self, x1, x2):
        v1 = self.linear(x1)
        v2 = torch.sigmoid(v1)
        v3 = v2 * v1
        return v3, x2

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 5)
x2 = torch.randn(3, 5, dtype=torch.int)
__output__, __output_1__ = m(x1, x2)

