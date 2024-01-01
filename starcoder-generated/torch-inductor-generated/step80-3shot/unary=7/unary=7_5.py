
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(64, 128)

    def forward(self, x1):
        y1 = self.linear(x1)
        y2 = y1 * torch.clamp(torch.clamp(y1, 0, 6) + 3, 0, 6)
        y3 = y2 / 6;
        return y3

# Initializing the model
m = Model()

