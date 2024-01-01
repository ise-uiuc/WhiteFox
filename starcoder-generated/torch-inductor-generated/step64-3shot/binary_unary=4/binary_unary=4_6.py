
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.linear = nn.Linear(128, 1000)
 
    def forward(self, x1, other):
        t1 = self.linear(x1)
        t2 = t1 + other
        t3 = F.relu(t2)
        return t3

# Initializing the model
m = Model()

# Inputs to the model
x2 = torch.randn(16, 128)
other = torch.randn(16, 1000)
