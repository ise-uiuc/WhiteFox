
class Model(torch.nn.Module):
    def __init__(self, other):
        super().__init__()
        self.linear = torch.nn.Linear(5, 4)
 
    def forward(self, x1, x2):
        t1 = self.linear(x1)
        t2 = t1 + x2
        t3 = F.relu(t2)
        return t3

# Initializing the model
m = Model(other = torch.randn(1, 4))

# Inputs to the model
x1 = torch.randn(1, 5)
x2 = torch.randn(1, 4)
