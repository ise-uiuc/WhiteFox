
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.features = torch.nn.Linear(3, 64)
 
    def forward(self, x1):
        v1 = self.features(x1)
        v2 = F.relu(v1)
        return v2

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 3)
