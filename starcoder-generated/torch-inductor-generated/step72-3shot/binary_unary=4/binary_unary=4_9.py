
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(3, 8)
 
    # Note that other is passed as a keyword argument
    def forward(self, x1, other):
        t1 = self.linear(x1)
        t2 = t1 + other
        t3 = nn.functional.relu(t2)
        return t3

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 3)
other = torch.randn(1, 8)
