 - linear+relu+linear
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = torch.nn.Linear(i, o1)
        self.linear2 = torch.nn.Linear(o1, o2)
 
    def forward(self, x):
        n = self.linear1(x)
        return self.linear2(n)
 
# Initializing the model
m = Model()

# Inputs to the model
x = torch.randn(b, i)
