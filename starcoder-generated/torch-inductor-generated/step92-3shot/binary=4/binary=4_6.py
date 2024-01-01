
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(5, 2048)
 
    def forward(self, x1):
        v1 = self.linear(x1)
        t2 = v1 + t1
        return t2

# Initializing the model
m = Model()

# Inputs to the model
t1 = torch.randn(1, 1, 1, 5)
x1 = torch.randn(1, 5, 1, 1)
