
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = torch.nn.Linear(32, 16, bias=True)
 
    def forward(self, x1):
        x2 = self.fc(x1)
        x3 = x2 + x1
        return x3

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(4, 32)
