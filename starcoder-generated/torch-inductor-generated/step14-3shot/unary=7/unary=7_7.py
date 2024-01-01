
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = torch.nn.Linear(3, 8)
 
    def forward(self, x1):
        l1 = self.fc(x1)
        l2 = l1 * torch.clamp(torch.clamp(l1, min=0), max=6) + 3
        l3 = l2 / 6
        return l3

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
