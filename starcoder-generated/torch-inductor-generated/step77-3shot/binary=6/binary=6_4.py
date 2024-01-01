
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(128, 9)
 
    def forward(self, x1, w1, b1, x2, w2, b2):
        v1 = self.linear(x1)
        v2 = v1 - x2
        return v2

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 64)
w1 = torch.randn(64, 9)
b1 = torch.randn(9)
x2 = torch.randn(1, 64)
w2 = torch.randn(64, 9)
b2 = torch.randn(9)
