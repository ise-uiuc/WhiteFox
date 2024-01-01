
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(16, 8, bias=False)
        self.other = torch.tensor([1, 2, 3, 4, 5, 6, 7, 8]).float().view(1, 2, 4)
    
    def forward(self, x1):
        v1 = self.linear(x1)
        v2 = v1 - self.other
        return v2

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 16, 32)
