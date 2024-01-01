
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(32 * 32, 64 * 32)
 
    def forward(self, x1):
        x1 = self.linear(x1)
        x2 = x1 + 1
        x3 = torch.relu(x2)
        return x3

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 32 * 32)
