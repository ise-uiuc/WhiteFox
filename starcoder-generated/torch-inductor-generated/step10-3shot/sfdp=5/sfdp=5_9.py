
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.key = torch.nn.Linear(3, 16)
 
    def forward(self, x1):
        v1 = self.key(x1)
        return v1

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 3)
