
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_1 = torch.nn.Linear(360, 144)
        self.linear_2 = torch.nn.Linear(48, 144)
    def forward(self, x1):
        v1 = self.linear_1(x1)
        v2 = self.linear_2(v1)
        v3 = torch.cat((v1, v2), 1)
        return v3

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 144)
v4 = torch.randn(1, 48)
