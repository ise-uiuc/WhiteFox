 for optimizing the cat
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.b = torch.nn.BatchNorm2d(1)

    def forward(self, x1):
        v1 = torch.cat([x1.unsqueeze(2), x1.unsqueeze(2)], 2)
        v2 = v1.view(3, 2) 
        v3 = self.b(v2)
        return v3

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 2, 2)
