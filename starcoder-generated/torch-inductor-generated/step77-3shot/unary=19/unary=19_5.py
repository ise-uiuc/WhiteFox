
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(32, 32)
 
    def forward(self, x1):
        v1 = self.linear(x1)
        y1 = torch.sigmoid(v1)
        return y1

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(8, 32)
