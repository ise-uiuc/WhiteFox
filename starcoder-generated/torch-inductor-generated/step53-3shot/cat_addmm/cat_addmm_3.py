
class Model(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        x = x + torch.mean(2 * torch.abs(x) - (x - x))
        x = x + torch.var(2 * torch.abs(x) - (x - x))
        return x
# Inputs to the model
x = torch.randn(2, 2)
