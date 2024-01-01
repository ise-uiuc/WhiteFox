
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(10, 10)
 
    def forward(self, xx):
        v1 = self.linear(xx)
        v2 = v1 + torch.full(size=(10,), fill_value=60, dtype=torch.float64)
        v3 = torch.nn.functional.relu(v2)
        return v3

# Initializing the model
m = Model()

# Inputs to the model
xx = torch.randn(10, 10)
