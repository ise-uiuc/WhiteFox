
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(3, 8, bias=True)
 
    def forward(self, x2):
        v4 = self.linear(x2)
        v5 = v4 - 0.5
        v7 = torch.relu(v5)
        return v7

# Initializing the model
m = Model()

# Inputs to the model
x2 = torch.randn(1, 3)
