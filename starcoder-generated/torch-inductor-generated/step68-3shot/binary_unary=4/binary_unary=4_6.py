
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(64 * 64, 32)
 
    def forward(self, x1, **kwargs):
        v1 = self.linear(x1)
        v2 = v1 + kwargs['other']
        v3 = v2.relu()
        return v3

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 64 * 64)
other = torch.randn(1, 32)
