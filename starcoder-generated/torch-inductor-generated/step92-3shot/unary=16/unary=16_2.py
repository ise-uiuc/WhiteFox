
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(1, 100)
 
    def forward(self, x1):
        v1 = self.linear(x1)
        v1 = F.relu(v1)
        return v1

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 1)
