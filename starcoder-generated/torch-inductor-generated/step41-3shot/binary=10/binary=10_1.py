
class Model(torch.nn.Module):
    def __init__(self, t5):
        super().__init__()
        self.linear = torch.nn.Linear(8, 16)
        self.other = t5
    
    def forward(self, x1):
        v1 = self.linear(x1)
        v2 = torch.add(v1, self.other)
        return v2

# Initializing the model
t5 = torch.randn(1, 16)
m = Model(t5)

# Inputs to the model
x1 = torch.randn(1, 8)
