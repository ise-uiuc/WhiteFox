
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(64, 3)
 
    def forward(self, x1, min_op=0.5, max_op=1.5):
        v1 = self.linear(x1)
        v2 = torch.clamp_min(v1, min_op)
        v3 = torch.clamp_max(v2, max_op)
        return v3

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 64)
x2 = torch.randn(1, 16)
