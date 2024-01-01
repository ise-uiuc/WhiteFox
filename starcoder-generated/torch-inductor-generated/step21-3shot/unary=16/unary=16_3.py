
class Model(torch.nn.Module):
    def __init__(self, size1, size2):
        super().__init__()
        self.linear = torch.nn.Linear(size1, size2)
 
    def forward(self, x1):
        v1 = self.linear(x1)
        v2 = torch.relu(v1)
        return v2

# Initializing the model
__size1__ = 64
__size2__ = 32
__model__ = Model(__size1__, __size2__)

# Initializing input of the model
x1 = torch.randn(1, __size1__)
