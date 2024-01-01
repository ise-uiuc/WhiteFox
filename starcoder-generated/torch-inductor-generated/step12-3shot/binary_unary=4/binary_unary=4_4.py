
class Model(torch.nn.Module):
    def __init__(self, other):
        super().__init__()
        self.linear = torch.nn.Linear(32, 64)
    
    def forward(self, x1):
        t1 = self.linear(x1)
        t2 = t1 + other
        t3 = torch.relu(t2)
        return t3

# Inputs to the model
x1 = torch.randn(64, 32)
other = torch.rand(64, 32)
