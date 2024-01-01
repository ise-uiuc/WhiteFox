
class Model(torch.nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.fc = torch.nn.Linear(dim, dim)
 
    def forward(self, x):
        v = self.fc(x)
        v = v + x
        v = F.relu(v)
        return v

# Initializing the model
m = Model(100)

# Inputs to the model
x = torch.randn(1, 100)
