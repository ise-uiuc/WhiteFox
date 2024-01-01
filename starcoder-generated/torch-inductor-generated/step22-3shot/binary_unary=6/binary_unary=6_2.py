
class Model(torch.nn.Module):
    def __init__(self, other):
        super().__init__()
        self.other = other
    
    def forward(self, x2):
        v1 = x2 + self.other
        v2 = torch.nn.functional.relu(v1)
        return v2

# Initializing the model
m = Model(other=torch.tensor(1))

# Inputs to the model
x2 = torch.randn(1, 16, 16, 16)
