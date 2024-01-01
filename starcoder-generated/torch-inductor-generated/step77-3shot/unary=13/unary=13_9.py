
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(20, 8, bias=False)
 
    def forward(self, x1):
        v0 = self.linear(x1)
        v1 = torch.sigmoid(v0)
        v2 = v0 * v1
        return v2

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(10, 20)
print("x1:", x1)
