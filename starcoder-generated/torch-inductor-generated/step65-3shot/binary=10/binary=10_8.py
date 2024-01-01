
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(128, 256)
 
    def forward(self, x1):
        x2 = self.linear(x1)
        x3 = x2 + x1
        x3 = x3.permute(0, 2, 1)
        return x3

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(64, 128)
