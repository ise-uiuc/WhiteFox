
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(3, 10)

    def forward(self, x1):
        out = self.linear(x1)
        out = out + 5.5
        out = torch.nn.ReLU()(out)
        return out

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
