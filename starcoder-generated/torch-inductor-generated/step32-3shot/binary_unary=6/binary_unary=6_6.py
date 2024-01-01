 - Basic
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(1000, 1000)
 
    def forward(self, x1):
        t1 = self.linear(x1)
        return t1

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(128, 1000)
