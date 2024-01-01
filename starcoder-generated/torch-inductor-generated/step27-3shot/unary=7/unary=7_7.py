
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = torch.nn.Linear(10, 10)
 
    def forward(self, x1):
        v1 = self.fc1(x1)
        v2 = v1 * torch.clamp(v1 + 3.0, 0.0, 6.0)
        v3 = v2 / 6.0
        return v3

# Initializing the model
m = Model()

# Input to the model
x1 = torch.randn(1, 10)
