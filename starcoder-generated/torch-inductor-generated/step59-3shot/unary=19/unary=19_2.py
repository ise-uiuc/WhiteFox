
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.Linear = torch.nn.Linear(3, 8)
 
    def forward(self, x1):
        v1 = self.Linear(x1)
        v2 = torch.sigmoid(v1)
        return v2

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
