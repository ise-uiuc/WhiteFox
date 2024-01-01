
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = torch.nn.Linear(32, 16)
 
    def forward(self, x1, x2 = torch.randn(1, 16)):
        v1 = self.l1(x1)
        v2 = v1 + x2
        v3 = torch.relu(v2)
        return v3

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 32)
