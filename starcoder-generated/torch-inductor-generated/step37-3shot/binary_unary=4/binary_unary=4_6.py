
class Model(torch.nn.Module):
    def forward(self, x1, other=torch.nn.Parameter(torch.ones(1, dtype=torch.float))):
        v1 = torch.nn.Linear(10, 5)(x1)
        v2 = v1 + other
        v3 = torch.nn.functional.relu(v2)
        return v3

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 10)
