
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(16, 8)
        self.other = torch.rand(8)
 
    def forward(self, x1):
        x2 = self.linear(x1)
        x3 = x2 - self.other
        x4 = F.relu(x3)
        return x4

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(4, 16)
