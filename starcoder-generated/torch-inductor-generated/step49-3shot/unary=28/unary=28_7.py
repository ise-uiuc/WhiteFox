
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.norm = torch.nn.BatchNorm2d(3)
 
    def forward(self, x1):
        v1 = self.norm(x1)
        return v1

# Initializing the model
min_value = 0.2
max_value = 0.3
m = Model()

# Inputs to the model
x2 = torch.randn(1, 3, 16, 16)
