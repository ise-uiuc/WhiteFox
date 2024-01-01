
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Linear(3, 8, bias=True)
 
    def forward(self, x1):
        return self.conv(x1)

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(12, 3)
