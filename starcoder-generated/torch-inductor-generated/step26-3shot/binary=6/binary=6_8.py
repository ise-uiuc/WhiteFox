
linear1 = torch.nn.Linear(4, 8, bias=False)
linear2 = torch.nn.Linear(8, 128, bias=False)
linear3 = torch.nn.Linear(128, 128, bias=False)
linear4 = torch.nn.Linear(128, 64, bias=False)
linear5 = torch.nn.Linear(64, 10, bias=False)

class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x1):
        v1 = linear1(x1)
        v2 = v1 - x1
        v3 = linear2(v1)
        v4 = linear3(v3 + v2)
        v5 = linear4(v3 + v2) + v4
        v6 = linear5(v5)
        return v6

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 4)
