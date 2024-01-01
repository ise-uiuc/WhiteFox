
class Model(torch.nn.Module):
    def __init__(self, other):
        super().__init__()
        self.linear = torch.nn.Linear(28, 10)
        self.other = torch.nn.Parameter(other)

    def forwark(self, x2):
        v1 = self.linear(x2)
        v2 = v1 + self.other
        v3 = torch.nn.functional.relu(v2)
        return v3

# Initializing the model with other as a keyword argument
m = Model(torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]))

# Initializing the model with other as a positional argument
m = Model(torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]).unsqueeze(0).unsqueeze(-1).unsqueeze(-1))

# Inputs to the model
x2 = torch.randn(1, 28, 1, 1)
