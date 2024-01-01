
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc2 = torch.nn.Linear(12, 16)
 
    def forward(self, x1, x2):
        v1 = self.fc2(x1)
        return v1 + x2

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(4, 12)
x2 = torch.randn(4, 16)
