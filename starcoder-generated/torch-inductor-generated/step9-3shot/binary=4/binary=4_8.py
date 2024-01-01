
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(13, 7)
 
    def forward(self, x1, x2):
        x3 = self.linear(x1)
        x4 = x3 + x2
        return x4

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 7, 11)
x2 = torch.randn(1, 7, 11)
