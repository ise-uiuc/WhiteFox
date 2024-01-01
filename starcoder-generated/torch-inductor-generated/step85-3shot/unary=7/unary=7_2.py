
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(3, 64, bias=True)
        
    def forward(self, x1):
        y1 = self.linear(x1)
        y2 = y1 * torch.clamp(min=0, max=6, y3 + 3)
        y3 = y2 / 48
        return y3

# Initializing the model
m = Model()

# Inputs to the model
__input__ = torch.randn(1, 3, 64, 64)
