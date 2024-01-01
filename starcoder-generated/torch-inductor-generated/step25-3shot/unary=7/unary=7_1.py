
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = torch.nn.Linear(6, 8)
        self.linear2 = torch.nn.Linear(8, 16)
 
    def forward(self, x1):
        y1 = self.linear1(x1)
        y2 = y1 * torch.clamp(y1, 0, 6)
        y3 = y2 / 6
        return y3

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(4, 6)
