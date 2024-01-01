
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(8, 4)
 
        self.other = torch.tensor([[81, -58, 49, -60]]).t()
 
    def forward(self, x1):
        v1 = self.linear(x1)
        v2 = v1 - self.other
        return v2

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 8)
