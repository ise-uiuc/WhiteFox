
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 6, 5, stride=1, padding=2)
        self.params = torch.nn.Linear(10, 6)
        
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = v1 + self.params
        v3 = torch.relu(v2)
        return v3

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 3, 7, 4)
