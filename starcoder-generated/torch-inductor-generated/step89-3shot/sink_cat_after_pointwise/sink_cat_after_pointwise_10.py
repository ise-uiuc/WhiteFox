
class Model(torch.nn.Module):
    # The only change is to specify a non-standard dimension
    def __init__(self):
        super().__init__()
    def forward(self, x):
        return torch.cat([x, x], dim=0).view(2, -1).tanh()
# Inputs to the model
x = torch.randn(2, 3, 4)
