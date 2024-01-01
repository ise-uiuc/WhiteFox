
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x1, x2):
        l = torch.nn.Linear(10, 10)
        out1 = l(x1)
        out2 = torch.add(out1, x2)
        out3 = torch.nn.functional.relu(out2)
        return out3

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 10)
x2 = torch.randn(1, 10)
