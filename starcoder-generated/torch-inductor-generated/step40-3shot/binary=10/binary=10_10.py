
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(64 * 64 * 3, 8, bias=True)
 
    def forward(self, x1, add):
        v1 = self.linear(x1)
        v2 = v1 + add
        return v2

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
add = torch.randn(1, 3, 64, 64)
