
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = torch.nn.Linear(32, 8)
 
    def forward(self, x, *, other):
        v1 = self.fc(x)
        v2 = v1 + other
        return v2

# Initializing the model
m = Model()

# Inputs to the model
x = torch.randn(2, 32)
other = torch.randn(8)
