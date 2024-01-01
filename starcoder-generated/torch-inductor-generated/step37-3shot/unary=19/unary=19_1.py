
class Model(torch.nn.Module):
    def __init__(self, size):
        super().__init__()
        self.fc = torch.nn.Linear(size, 1)
 
    def forward(self, x1):
        v1 = self.fc(x1)
        v2 = torch.sigmoid(v1)
        return v2

# Initializing the model
size=32
m = Model(size)

# Inputs to the model
x1 = torch.randn(1, size)
