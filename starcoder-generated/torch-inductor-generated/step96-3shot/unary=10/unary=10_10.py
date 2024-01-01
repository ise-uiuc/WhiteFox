
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(64, 128)
 
    def forward(self, x):
        t1 = self.linear(x)
        t2 = t1 + 3
        t3 = torch.clamp_min(t2, 0)
        t4 = torch.clamp_max(t3, 6)
        t5 = t4 / 6
        return t5

# Initializing the model
m = Model()

# Input to the model
x = torch.randn(1, 64)
