
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(40, 50)
 
    def forward(self, x1):
        t1 = self.linear(x1)
        t2 = torch.clamp_min(t1, -0.1)
        t3 = torch.clamp_max(t2, 1.1)
        return t3

# Initializing the model
m = Model()

# Inputs to the model
torch.randn(1, 40)
