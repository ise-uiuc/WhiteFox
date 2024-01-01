
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        torch.nn.init.zeros_(self.layers.weight)
        torch.nn.init.zeros_(self.layers.bias)
    
    def forward(self, x):
        x = self.layers(x)
        x1 = torch.cat((x[0], x[1]), dim=1)
        x2 = torch.cat((x1[0], x1[1]), dim=1)
        x3 = torch.cat((x2[0], x2[1]), dim=1)
        return x3
# Inputs to the model
x = torch.randn(2, 2)
