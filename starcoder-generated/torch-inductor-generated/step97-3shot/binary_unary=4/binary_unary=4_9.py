
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(64, 128)
 
    def forward(self, data, other):
        v1 = self.linear(data)
        v2 = v1 + other
        v3 = v2.relu()
        return v3

# Initializing the model
m = Model()

# Inputs to the model
data = torch.randn(8, 64)
other = torch.randn(8, 128)
