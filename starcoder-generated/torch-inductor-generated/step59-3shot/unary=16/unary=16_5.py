
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(3072, 4096)
 
    def forward(self, x1):
        y = self.linear(x1)
        return y

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 3072)
