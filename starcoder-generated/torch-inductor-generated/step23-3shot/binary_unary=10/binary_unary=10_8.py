
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = torch.nn.Linear(1000, 4096)
        self.linear2 = torch.nn.Linear(4096, 4096)
        self.linear3 = torch.nn.Linear(4096, 7)
 
    def forward(self, x1, x2, x3):
        t1 = self.linear1(x1)
        t2 = self.linear2(x2)
        t3 = self.linear3(x3)
 
        cat1 = torch.cat((t1, t2), 1)
        cat2 = torch.cat((cat1, t3), 1)
 
        return cat2

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1000)
x2 = torch.randn(1000)
x3 = torch.randn(7)
