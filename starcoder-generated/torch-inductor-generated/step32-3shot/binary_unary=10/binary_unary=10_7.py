
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(3, 1)
 
    def forward(self, x1, other):
        t1 = self.linear(x1)
        t2 = t1 + other
        t3 = torch.relu(t2)
        return t3

# Initializing the model
model1 = Model()

# Inputs to the model
x1 = torch.ones(1, 3)
other = torch.rand(1, 1)
