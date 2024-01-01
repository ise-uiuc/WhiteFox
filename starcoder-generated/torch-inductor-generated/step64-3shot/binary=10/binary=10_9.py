
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(16, 32)
 
    def forward(self, x1, **kwargs):
        v1 = self.linear(x)
        v2 = v1 + kwargs['other']
        return v2

# Initializing the model
m = Model()

# Inputs to the model
x = torch.randn(1, 16)
other = torch.randn(1, 32)
