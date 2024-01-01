
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(64, 16)
 
    def forward(self, x2):
        v0 = x2.shape[0]
        v1 = x2.view(v0, -1)
        v2 = self.linear(v1)
        return v2 - 5

# Initializing the model
m = Model()

# Inputs to the model
x2 = torch.randn(1, 64)
