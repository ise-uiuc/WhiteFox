
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.unfold = torch.nn.Unfold(1, 3, 1, 1)
 
    def forward(self, x):
        v1 = self.unfold(x)
        v2 = v1.split(2)
        v3 = torch.cat(v2)
        return v3

# Initializing the model
m = Model()

# Inputs to the model
x = torch.randn(1, 3, 50)
