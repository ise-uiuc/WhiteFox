
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = torch.nn.Linear(2, 2)
        self.linear2 = torch.nn.Linear(2, 2)
        self.linear3 = torch.nn.Linear(2, 2)

    def forward(self, x):
        v1 = torch.nn.functional.linear(x, self.linear1.weight, self.linear1.bias)
        v2 = torch.add(x, x)
        v3 = torch.nn.functional.linear(v2, self.linear2.weight, self.linear2.bias)
        v4 = torch.nn.functional.linear(torch.cat((v1, v3), dim=1), self.linear3.weight, self.linear3.bias)
        return v4

# Initializing the model
m = Model()

# Inputs to the model
x = torch.randn(1, 2, 2)
