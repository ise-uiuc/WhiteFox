
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(23, 8)
 
    def forward(self, x1, **kwargs):
        other = kwargs['other']
        v1 = self.linear(x1)
        v2 = v1 + other
        v3 = nn.ReLU()(v2)
        return v3

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 23)
other = torch.randn(1, 8)
