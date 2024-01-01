
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(100, 100)
 
    def forward(self, x1, other):
        v1 = self.linear(x1)
        v2 = v1 - other
        v3 = F.relu(v1)
        return v2, v3

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(256, 100)
other = torch.randn(256, 100)
__output1__, __output2__ = m(x1, other)

