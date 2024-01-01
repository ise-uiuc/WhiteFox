
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = torch.nn.Linear(100, 10)
        self.linear2 = torch.nn.Linear(10, 1)
 
    def forward(self, x1):
        t1 = self.linear1(x1)
        t2 = t1 + other
        y1 = self.linear2(t2)
        return y1

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(16, 100)
other = torch.randn(16, 10)
