
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = torch.nn.Linear(8, 2)
 
    def forward(self, x1):
        v1 = torch.linear(x1)
        va = torch.sigmoid(v1)
        return va, v1

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(2, 8)
__output__, __output2__ = m(x1)

