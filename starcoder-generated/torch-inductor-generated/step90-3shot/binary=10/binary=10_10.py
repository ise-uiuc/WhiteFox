
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = torch.nn.Linear(in_features=8, out_features=16, bias=False)
        self.linear2 = torch.nn.Linear(in_features=16, out_features=8, bias=False)

    def forward(self, x):
        v1 = self.linear1(x)
        v2 = v1 + other
        v3 = self.linear2(v2)
        return v3

# Initializing the model
m = Model()

# Inputs to the model
x = torch.randn(1, 8)
