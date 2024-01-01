
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(3, 8)
 
    def forward(self, x1):
        x2 = self.linear(x1)
        x3 = torch.sigmoid(x2)
        return x3

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 3)
