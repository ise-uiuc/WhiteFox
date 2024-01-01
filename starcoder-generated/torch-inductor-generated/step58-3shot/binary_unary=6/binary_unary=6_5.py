
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear0 = torch.nn.Linear(64, 1)

    def forward(self, x1):
        t1 = self.linear0(x1)
        t2 = t1 - 3.1
        t3 = torch.relu(t2)
        return t3

# Initializing the model
model = Model()

# Inputs to the model
x1 = torch.randn(1, 64)
