
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(100, 100)
 
    def forward(self, x1, x2):
        x = torch.cat((x1, x2), 1)
        t1 = self.linear(x)
        t2 = t1 - x2
        t3 = F.relu(t2)
        return t3

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 100)
x2 = torch.randn(1, 100)
