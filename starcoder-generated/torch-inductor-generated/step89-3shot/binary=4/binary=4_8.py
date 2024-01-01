
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = torch.nn.Linear(64, 32, stride=4, padding=1)
 
    def forward(self, x1, **kwargs):
        t2 = self.linear1(x1)
        out = t2 + kwargs['other_tensor']
        return out

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 64, 23, 23)
other_tensor = torch.ones(1, 32, 23, 23)
m(x1, other_tensor=other_tensor)

