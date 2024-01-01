
class Model(torch.nn.Module):    
    def __init__(self):
        super().__init__()

    def forward(self, x1, x2):
        x1 = x1.view((x1.size()[0], -1))
        x2 = x2.view((x2.size()[0], -1))
        x = torch.cat([x1, x2], dim=1)
        return x

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 25, 32, 32)
x2 = torch.randn(1, 50, 16, 16)
