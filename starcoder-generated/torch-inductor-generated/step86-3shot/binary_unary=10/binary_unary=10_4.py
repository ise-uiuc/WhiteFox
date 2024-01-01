
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(1024, 10)
 
    def forward(self, x1, x2):
        x3 = torch.tanh(x1 + x2)
        v1 = self.linear(x3)
        v2 = v1 + x2
        v3 = torch.relu(v2)
        return v3, v2

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 1024)
x2 = torch.randn(1, 10)
