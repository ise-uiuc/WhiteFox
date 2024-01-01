
class Model(torch.nn.Module):
    def __init__(self, w):
        super().__init__()
        self.w = w
 
    def forward(self, x1):
        v1 = torch.mm(x1, self.w.t())
        v2 = v1 - x1
        return v1

# Initializing the model
m = Model(torch.randn((64, 10)))

# Inputs to the model
x1 = torch.randn(64, 10)
