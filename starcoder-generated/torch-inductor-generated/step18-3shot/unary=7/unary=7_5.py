
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(3, 8)
 
    def forward(self, x1):
        q1 = self.linear(x1)
        q2 = q1 * torch.clamp_min_max(min=0, max=6, q1 + 3)
        q3 = q2 / 6
        return q3

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 3)
