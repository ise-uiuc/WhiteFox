
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_layer = torch.nn.Linear(8, 16, bias=False)
 
    def forward(self, x1):
        t1 = self.linear_layer(x1)
        t2 = t1 + 3
        t3 = torch.clamp(t2, 0, 6)
        return t3 / 6

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 8)
