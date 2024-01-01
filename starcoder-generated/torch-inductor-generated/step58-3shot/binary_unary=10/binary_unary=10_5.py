
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(3, 3)
 
    def forward(self, x1):
        v1 = self.linear(x1)
        v2 = torch.empty_like(v1, requires_grad=True)
        v2.random_(9)
        v3 = v1 + v2
        v4 = torch.nn.functional.relu(v3)
        return v4

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 3)
