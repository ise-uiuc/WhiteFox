
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        x = x.expand(x.shape[0], 1 * x.shape[1]).contiguous()
        return torch.cat((x, x), dim=1)
# Inputs to the model
x = torch.randn(1, 2)
