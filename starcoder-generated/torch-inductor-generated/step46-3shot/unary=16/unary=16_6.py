
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(64, 16)
 
    def forward(self, x1):
        v2 = F.relu(self.linear(x1))
        return v2

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(2, 64)
