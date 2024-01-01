
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(20, 32, bias=True)
        self.relu = torch.nn.ReLU()
 
    def forward(self, x1, other=None):
        v2 = self.linear(x1)
        if other is None:
            v3 = v2
        else:
            v3 = v2 + other
        v4 = self.relu(v3)
        return v4

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 20, stride=1, padding=1)
