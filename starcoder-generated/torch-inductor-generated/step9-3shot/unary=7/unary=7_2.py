
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(8, 8)

    def forward(self, x1):
        w1 = self.linear(x1)
        w2 = w1 * torch.clamp(torch.min(w1), torch.max(w1), w1 + 3)
        w3 = w2 / 6
        return w3

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 8)
