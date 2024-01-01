
class Model(torch.nn.Module):
    def __init__(self, t1):
        super().__init__()
        self.t1 = t1
 
    def forward(self, x1):
        v1 = self.t1(x1)
        v2 = v1 + x1
        return v2

# Initializing the model
t1 = torch.randn(2, 3)
m = Model(t1)

# Inputs to the model
x1 = torch.randn(1, 2, 3)
