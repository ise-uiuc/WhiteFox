
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear0 = torch.nn.Linear(20, 20)
        self.linear1 = torch.nn.Linear(20, 20)

    def forward(self, x1, x2):
        v1 = self.linear0(x1)
        v2 = self.linear1(x2)
        v3 = v1 + v2
        return v3

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 20)
x2 = torch.randn(1, 20)
