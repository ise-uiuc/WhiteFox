
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(512, 64)
 
    def forward(self, x1, t):
        v1 = self.linear(x1)
        v2 = v1 + t
        v3 = torch.relu(v2)
        return v3

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 512)
t = torch.randn(1, 64)
