
class Model(torch.nn.Module):
    def __init__(self, size):
        super().__init__()
        self.linear = torch.nn.Linear(10, size, 1, stride=1, bias=False)
 
    def forward(self, x1, x2):
        v1 = self.linear(x1)
        v2 = v1 + x2
        return v2

# Initializing the model
m = Model(30)

# Inputs to the model
x1 = torch.randn(1, 10)
x2 = torch.randn(1, 30)
__output1__, __output2__ = m(x1, x2)

