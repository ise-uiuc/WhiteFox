
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = torch.nn.Linear(64, 8)
        self.linear2 = torch.nn.Linear(8, 10)
 
    def forward(self, x1, *, other=0):
        y2 = self.linear2(self.linear1(x1))
        return (y2 + other).softmax(dim=1)

# Initializing the model
m = Model()

# Inputs to the model (x1 is specified, other is used in its place)
x1 = torch.randn(1, 64)
