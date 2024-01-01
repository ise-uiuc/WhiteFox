
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(3, 6)
 
    def forward(self, x1):
        v1 = self.linear(x1)
        t1 = torch.clamp_min(v1, min_value=0.1) # Clamps the minimum output of the linear transformation to 0.1
        t2 = torch.clamp_max(t1, max_value=0.4) # Clamps the maximum output of t1 to 0.4
        return t1, t2

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 3)
__output1__, __output2__ = m(x1)
