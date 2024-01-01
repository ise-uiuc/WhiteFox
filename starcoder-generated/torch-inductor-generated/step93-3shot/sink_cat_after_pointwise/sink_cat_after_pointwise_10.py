
class NewModel(torch.nn.Module):
    def __init__(self, shape):
        super().__init__()
        self.x = nn.Parameter(torch.randn(shape))
    def forward(self, x):
        y = torch.cat(self.x, dim=1)
        y = torch.relu(y)
        return y
# Inputs to the model
x = torch.randn(1, 2, 3)
