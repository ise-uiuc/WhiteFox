
class Model1(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(1280, 1000)
        self.linear1 = torch.nn.Linear(1000, 500)

    def forward(self, x1):
        v1 = self.linear(x1)
        v2 = v1 + self.linear1(v2)
        return v2

# Initializing the model
m = Model1()

# Inputs to the model
x1 = torch.randn(3, 1280)
