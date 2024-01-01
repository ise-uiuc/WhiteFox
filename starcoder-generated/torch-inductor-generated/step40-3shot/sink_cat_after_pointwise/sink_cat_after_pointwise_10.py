
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        return (torch.cat([x, x], dim=1).view(x.shape[0], -1).leaky_relu(negative_slope=0.2) + torch.cat([x, x], dim=1).view(x.shape[0], -1).leaky_relu(negative_slope=0.05)).sum()
# Inputs to the model
x = torch.randn(2, 3, 4)
