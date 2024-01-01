
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.model_dim = 16
        self.n_layers = 2
        self.conv = torch.nn.Conv2d(self.model_dim, self.model_dim, 3, padding=1)
    def forward(self, x1):
        for _ in range(self.n_layers):
            v1 = self.conv(x1)
            x1 = x1 + v1
            x1 = torch.relu(x1)
        return x1
# Inputs to the model
x1 = torch.randn(1, 16, 100, 100)
