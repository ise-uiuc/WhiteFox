
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        y = torch.arange(24).reshape(2, 3, 4)
        if x.shape[0] > y.shape[0]:
            y = torch.stack([y]*x.shape[0])
        z = torch.cat([x, y], dim=1)
        z1 = z.view(z.shape[0], -1)
        z2 = torch.tanh(z1)
        return z2    
# Inputs to the model
x = torch.randn(2, 3)
