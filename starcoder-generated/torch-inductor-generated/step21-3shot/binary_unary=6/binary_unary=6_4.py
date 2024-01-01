
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(64, 128)
 
    def forward(self, x1, other):
        v1 = self.linear(x1)
        v2 = v1 - other
        v3 = F.relu(v1 - other)
        return v3


# Initializing the model
m = Model()
# Parameters of the model are initialized randomly

# Inputs to the model
x1 = torch.randn(3, 64)
other = 1.0
