
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(16, 1, bias=False)
        self.linear.weight = torch.nn.Parameter(torch.eye(16, 1) * 0.1)
 
    def forward(self, x1):
        t1 = self.linear(x1)
        t2 = t1 + x1
        t3 = torch.relu(t2)
        return t3

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(8, 16)
