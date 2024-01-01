
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(960, 512)
 
    def forward(self, x1, x2):
        v1 = self.linear(x1)
        v2 = v1.add(x2)
        v3 = F.relu(v2)
        return v3

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(4, 960)
x2 = torch.randn(4, 512)
