
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 70, bias=True)
 
    def forward(self, x1, other):
        t1 = self.linear(x1)
        t2 = t1 + other
        t3 = torch.nn.functional.relu(t2)
        return t3

# Initializing the model
other = torch.rand(70)
m = Model()

# Inputs to the model
x1 = torch.randn(1, 2)
