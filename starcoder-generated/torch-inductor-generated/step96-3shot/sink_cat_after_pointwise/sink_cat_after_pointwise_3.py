
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        x0 = x.clone()
        x1 = x0 + x0
        x2 = x1.sum(1)[:,None]
        x3 = x2.squeeze(1).tanh()
        x4 = torch.cat([x, x3], dim=1)
        return x4
# Inputs to the model
x = torch.randn(2, 3)
