
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(3, 4, bias=True)
        self.other = torch.nn.Parameter(torch.tensor([0.1, 0.2, 0.3, 0.7], dtype=torch.float32))
 
    def forward(self, x1):
        v1 = self.linear(x1)
        v2 = v1 - self.other
        v3 = torch.nn.functional.relu(v2)
        return v3

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 3)
