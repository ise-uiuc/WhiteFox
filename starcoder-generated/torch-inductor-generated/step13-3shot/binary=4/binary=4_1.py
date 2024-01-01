
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = torch.nn.Linear(16, 16)
        self.linear2 = torch.nn.Linear(16, 16)
    
    def forward(self, x1, **kwargs):
        x1 = F.relu(self.linear1(x1))
        x2 = self.linear2(x1)
        return x2

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 16)
x2 = torch.randn(1, 16)
