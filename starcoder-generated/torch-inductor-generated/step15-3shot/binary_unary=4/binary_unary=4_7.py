
class Model(torch.nn.Module):
    def __init__(self, other=torch.zeros((1, 64))):
        super().__init__()
        self.linear = torch.nn.Linear(64, 64)
 
    def forward(self, x1, **kwargs):
        kwargs = kwargs if kwargs is not None else {}
        v1 = self.linear(x1)
        v2 = v1 + kwargs['other']
        v3 = torch.relu(v2)
        return v3

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 64)
