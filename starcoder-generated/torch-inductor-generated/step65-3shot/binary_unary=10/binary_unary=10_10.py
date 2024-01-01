
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(5, 3)
 
    def forward(self, x1):
        v2 = self.linear(x1)
        v1 = v2 + torch.ones(v2.shape)
        v3 = torch.relu(v1)
        return v3

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 5)
