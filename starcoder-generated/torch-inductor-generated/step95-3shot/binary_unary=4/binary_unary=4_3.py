
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(16, 2, bias=False)
        self.other = torch.randn(2, 16)
 
    def forward(self, x1, **kargs):
        v1 = self.linear(x1)
        v2 = self.other + v1
        v3 = F.relu(v2)
        return v3

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 1, 32, 32)
