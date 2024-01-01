
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(18, 1)
        self.linear.weight.data.fill_(0.5) 

    def forward(self, x1):
        t1 = self.linear(x1)
        t2 = torch.sigmoid(t1)
        return t2

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 18)
