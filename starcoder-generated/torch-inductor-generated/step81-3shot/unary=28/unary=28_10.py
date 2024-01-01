
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(128, 256)

    def forward(self, x1):
        y1 = self.linear(x1)
        y2 = torch.clamp(y1, min=0.)
        return y2

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 64, 64)
