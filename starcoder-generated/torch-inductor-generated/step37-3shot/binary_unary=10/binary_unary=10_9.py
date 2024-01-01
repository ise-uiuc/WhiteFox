
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(30, 20)
 
    def forward(self, x1, x2, x3):
        v1 = self.linear(x1)
        v2 = v1 + x2
        v3 = torch.relu(v2)
        v4 = v3 + x3
        return v4

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 30)
x2 = torch.randn(1, 30)
x3 = torch.randn(1, 30)
