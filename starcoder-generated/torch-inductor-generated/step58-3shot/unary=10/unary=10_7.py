
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = torch.nn.Linear(3, 8, bias=False)
        self.linear2 = torch.nn.Linear(8, 8, bias=False)

    def forward(self, x):
        x = self.linear1(x)
        x = self.linear2(x)
        x = x + 3
        x = torch.clamp_max(torch.clamp_min(x, 0), 6)
        x = x / 6
        return x

# Initializing the model
m = Model()

# Inputs to the model
x = torch.randn(1, 3, 64, 64)
