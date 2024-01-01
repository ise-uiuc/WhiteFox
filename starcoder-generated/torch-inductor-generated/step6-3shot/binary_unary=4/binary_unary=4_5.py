
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(40, 40)
 
    def forward(self, x1, other=torch.tensor([1, 2, 3, 4], device="cpu", dtype=torch.float)):
        v1 = self.linear(x1)
        v2 = v1 + other
        v3 = torch.relu(v2)
        return v3

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 40)
