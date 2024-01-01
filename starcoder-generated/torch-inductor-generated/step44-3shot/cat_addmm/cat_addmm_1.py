
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Linear(256, 256)
    def forward(self, x):
        x = self.layers(x)
        x = torch.transpose(x, 0, 1)
        x = torch.matmul(x, x)
        return x
# Inputs to the model
x = torch.randn(20, 1, 256)
