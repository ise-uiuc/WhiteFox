
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(128, 1000)
 
    def forward(self, x1):
        y1 = self.linear(x1)
        y2 = y1 * torch.clamp(torch.tensor(0.0), torch.tensor(6.0), y1 + 3.0)
        y21 = y2 / 6.0
        return y21

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(2, 128)
