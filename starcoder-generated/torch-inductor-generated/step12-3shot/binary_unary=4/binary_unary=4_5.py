
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 7)
 
    def forward(self, x1, x2):
        v1 = self.linear(x1)
        v3 = torch.relu(v1 + x2)
        return v3

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 2)
__other__ = torch.randn(1, 7)
