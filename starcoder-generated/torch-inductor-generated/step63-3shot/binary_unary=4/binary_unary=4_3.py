
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(16, 32)
 
    def forward(self, x1):
        v1 = self.linear(x1)
        y = v1 + x2
        y = F.relu(y)
        return y

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 16)
x2 = torch.randn(1, 32)
