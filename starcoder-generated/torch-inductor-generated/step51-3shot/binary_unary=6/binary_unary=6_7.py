
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(10, 25)
 
    def forward(self, x1):
        t1 = self.linear(x1)
        t2 = t1 - 1
        t3 = torch.relu(t2)
        return t3

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(25, 10)
