
class Model(torch.nn.Module):
    def __init__(self, negative_slope=0.1):
        super().__init__()
        self.linear = torch.nn.Linear(32, 33)
        self.l = negative_slope

    def forward(self, x1):
        i1 = self.linear(x1)
        y1 = i1 > 0
        y2 = i1 * self.l
        y3 = torch.where(y1, y1, y2)
        return y3

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 32)
