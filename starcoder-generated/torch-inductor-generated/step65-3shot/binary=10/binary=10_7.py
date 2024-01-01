
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(16, 64)
 
    def forward(self, x1, **kwargs):
        v1 = self.linear(x1)
        v2 = v1 + kwargs['other']
        return v2

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(4, 4)

# Arguments for the model
other = torch.randn(4, 1)
new_m = m.eval()
