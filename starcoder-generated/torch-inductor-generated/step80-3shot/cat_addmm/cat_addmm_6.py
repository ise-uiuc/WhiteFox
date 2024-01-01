
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Linear(4, 4)
        self.conv = nn.Linear(4, 4)
    def forward(self, x):
        x = self.layers(x)
        x_clone = x.transpose(0, 1).reshape(-1)
        x_clone = self.conv(x_clone)
        x_clone = x_clone.view(4, 2).transpose(0, 1)
        x = x + x_clone
        return x
# Inputs to the model
x = torch.randn(2, 4)
