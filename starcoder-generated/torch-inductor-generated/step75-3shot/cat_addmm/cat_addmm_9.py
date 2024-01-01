
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Conv2d(1, 1, (1, 1))
        self.relu = nn.functional.relu
    def forward(self, x):
        x = self.layers(x)
        x = self.relu(x)
        x = x.permute((0, 2, 3, 1))
        x = x.reshape((-1, 1))
        return x
# Inputs to the model (2D tensor with 3 channels)
x = torch.randn(1, 3, 1, 1)
