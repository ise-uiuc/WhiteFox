
class Model(torch.nn.Module):
    def __init__(self, a):
        super().__init__()
        self.linear = torch.nn.Linear(in_features=224, out_features=256, bias=True)
 
    def forward(self, x2):
        z1 = self.linear(x2)
        z2 = torch.nn.functional.relu(z1)
        return z2

# Initializing the model
m = Model(a=256)

# Inputs to the model
x2 = torch.randn(256, 224)
