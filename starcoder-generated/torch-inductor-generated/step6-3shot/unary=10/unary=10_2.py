
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = torch.nn.Linear(128, 1)
 
    def forward(self, x1):
        a1 = self.fc(x1)
        a2 = a1 + 3
        a3 = torch.clamp_max(a2, 5)
        a4 = torch.clamp_min(a3, 0)
        a5 = a4 / 5
        return a4

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(2, 128)
