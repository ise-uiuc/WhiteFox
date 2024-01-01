
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(8, 16)
 
    def forward(self, x1, x2):
        v1 = self.linear(x1)
        v3 = x2 / self.linear(x1 + x2)
        return v1, v3

# Initializing the model
x1 = torch.randn(1, 8)
x2 = torch.randn(1, 8)
m = Model()

# Inputs to the model
__output1__, __output2__ = m(x1, x2)

