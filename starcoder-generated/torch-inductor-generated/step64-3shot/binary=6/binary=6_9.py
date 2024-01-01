
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = torch.nn.Linear(960, 256)
 
    def forward(self, x1):
        v1 = self.fc1(x1)
        v2 = v1 - 0.1
        return v2

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 960)
