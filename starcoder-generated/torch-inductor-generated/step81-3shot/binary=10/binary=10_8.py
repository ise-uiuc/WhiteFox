
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(1,1)
 
    def forward(self, x1, **kwargs):
        v1 = self.linear(x1)
        v2 = v1 + kwargs['other']
        return v2

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(7,3,3)
kwargs = {}
kwargs['other'] = torch.randn(7,3,1)
