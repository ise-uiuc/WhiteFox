
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(64, 64)
 
    def forward(self, x1, other):
        v1 = self.linear(x1)
        v2 = v1 + other
        return v2

# Initializing the model
input0 = torch.randn(1, 1, 64)
input1 = torch.randn(1, 1, 64)
m = Model()

# Inputs to the model
