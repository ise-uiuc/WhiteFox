
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Linear(3, 64)
    
    def forward(self, x2):
        v2 = self.conv(x2)
        v3 = torch.relu(v2)
        return v3

# Initializing the model
m2 = Model()
# Inputs to the model
x2 = torch.randn(1, 3)
