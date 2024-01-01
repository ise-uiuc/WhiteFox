
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(8, 1)

    def forward(self, x3: torch.Tensor, other):
        t1 = self.linear(x3)
        t2 = t1 + other
        t3 = torch.nn.functional.relu(t2)
        return t3

# Initializing the model
m = Model()

# Inputs to the model
x3 = torch.randn(1, 8)
