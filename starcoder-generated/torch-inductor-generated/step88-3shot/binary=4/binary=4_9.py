
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_ = torch.nn.Linear(3, 8)
 
    def forward(self, x1):
        v1 = self.linear_(x1)
        v2 = v1 + other
        return v2

# Initializing the model
m = Model()
with torch.no_grad():
    m.linear_.weight.copy_(torch.eye(3) * 0.1, non_blocking=True)
    m.linear_.bias.copy_(torch.ones(3) * 0.1, non_blocking=True)

# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
