
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = torch.nn.Linear(3, 15, False)
 
    def forward(self, x1):
        v1 = self.l1(x1)
        v2 = v1.clamp(min=0)
        return v2

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
