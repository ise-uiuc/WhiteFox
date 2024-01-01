
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(8, 4)
 
    def forward(self, x1, **kwargs):
        v0 = kwargs['other_tensor']
        v1 = self.linear(x1)
        v2 = v1 + v0
        v3 = F.relu(v2)
        return v3

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 8)
other_tensor = torch.randn(1, 4)
