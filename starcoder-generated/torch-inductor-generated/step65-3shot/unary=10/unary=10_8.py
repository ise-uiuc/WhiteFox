
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = torch.nn.Linear(1280, 640)
    
    def forward(self, x1):
        v1 = self.fc1(x1)
        v2 = v1 + 3
        v3 = torch.clamp(v2, 0, 6)
        return v3 / 6

# Initializing the model
m = Model()

# Input to the model
x1 = torch.randn(1, 1280)
