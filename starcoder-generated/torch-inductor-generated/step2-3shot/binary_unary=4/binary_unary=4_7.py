
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(in_features=4, out_features=3)
 
    def forward(self, x1, **kwargs):
        v1 = self.linear(x1)
        v2 = v1 + kwargs.get('other')
        v3 = torch.relu(v2)
        return v3

# Initializing the model
m1 = Model()

# Inputs to the model
x1 = torch.randn(1, 4)
