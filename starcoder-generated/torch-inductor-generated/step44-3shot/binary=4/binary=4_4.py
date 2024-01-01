
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = torch.nn.Linear(8, 8)
 
    def forward(self, x1):
        y1 = self.l1(x1).
        y2 = y1 + 1
        return y2

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 8)
