
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = torch.nn.Linear(3, 11)
        self.linear2 = torch.nn.Linear(11, 16)
 
    def forward(self, x1):
        t1 = self.linear1(x1)
        t2 = t1 + 1
        t3 = self.linear2(t2)
        return t3

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(20, 3)
