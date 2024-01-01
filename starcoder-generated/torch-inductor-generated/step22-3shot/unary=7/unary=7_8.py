
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(3, 64, bias=False)
 
    def forward(self, x1):
        v1 = self.linear(x1)
        v2 = self.linear(x1)
        v3 = v2 + 3
        v4 = torch.nn.functional.relu6(v3)
        v5 = v4 / 6
        return v5

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 3)
