
class Model(torch.nn.Module):
    def __init__(self, other):
        super().__init__()
        self.linear = torch.nn.Linear(1280, 256)
        self.other = torch.reshape(other, (1, 256, 1, 1))
 
    def forward(self, x1):
        v1 = self.linear(x1)
        v2 = v1 + self.other
        return v2

# Initializing the model
v_other = torch.tensor(42.0)
m = Model(v_other)

# Inputs to the model
x1 = torch.randn(1, 1280)
