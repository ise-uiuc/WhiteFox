
class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.op = torch.nn.Linear(1920, 10)
 
    def forward(self, x1, x2):
        v1 = self.op(x1)
        v2 = v1 + x2
        v3 = v2 * (v2 > 3).float()
        return v3

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 1920)
x2 = torch.randn(1, 10)
