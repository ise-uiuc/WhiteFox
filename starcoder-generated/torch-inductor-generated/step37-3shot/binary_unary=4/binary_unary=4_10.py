
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(51200, 1024)
 
    def forward(self, x1):
        v1 = self.linear(x1)
        __return t2__

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 51200)
t1 = torch.randn(1, 51200)
