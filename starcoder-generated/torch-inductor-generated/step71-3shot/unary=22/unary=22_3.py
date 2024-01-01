
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = torch.nn.Linear(256 * 256, 100)
        self.linear2 = torch.nn.Linear(100, 10)

    def forward(self, x1):
        t1 = self.linear1(x1)
        t2 = torch.tanh(t1)
        return t2

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 256, 256)
