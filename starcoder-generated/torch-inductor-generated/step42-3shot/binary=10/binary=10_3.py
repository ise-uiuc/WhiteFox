
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = torch.nn.Linear(3 * 32 * 32, 64)
 
    def forward(self, x, other):
        v1 = self.linear1(x)
        v2 = v1 + other
        return v2

# Initializing the model
m = Model()

# Inputs to the model
x = torch.randn(1, 3 * 32 * 32)
other = torch.randn(1, 64)
