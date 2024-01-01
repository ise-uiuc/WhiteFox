
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.other = Parameter(torch.randn(8))
 
    def forward(self, x1):
        v1 = torch.nn.functional.linear(x1, self.other)
        v2 = v1 + self.other
        v3 = torch.relu(v2)
        return v3

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 8)
