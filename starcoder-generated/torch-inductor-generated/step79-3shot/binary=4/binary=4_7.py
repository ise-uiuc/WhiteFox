
class AddLayer(torch.nn.Module):
    def forward(self, x1, x2):
        v1 = torch.nn.functional.linear(x1, weight=torch.rand(64, 3), bias=torch.rand(64))
        v2 = x1 + x2
        v3 = v2 + v1
        return v3
 
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.add_layer = AddLayer()
 
    def forward(self, x1, x2):
        v = self.add_layer(x1, x2)
        return v

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 64)
x2 = torch.randn(1, 3, 64, 64)
