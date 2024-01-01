
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(8, 8)
 
    def forward(self, x1):
        t1 = self.linear(x1)
        t2 = t1 + 3
        t3 = torch.clamp_min(t2, 0)
        t4 = torch.clamp_max(t3, 6)
        t5 = t4 / 6
        return t5

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 8)
