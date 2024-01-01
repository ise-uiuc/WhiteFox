
class Model(torch.nn.Module):
    def __init__(self, conv_dim):
        super().__init__()
        self.layers = [torch.nn.Conv2d(3, conv_dim, conv_dim), torch.nn.BatchNorm2d(conv_dim)]
        for _ in range(10):
            self.layers.append(torch.nn.ReLU())
            self.layers.append(torch.nn.Conv2d(conv_dim, conv_dim, conv_dim))
            self.layers.append(torch.nn.BatchNorm2d(conv_dim))
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
# Inputs to the model
x = torch.randn(1, 3, 6, 6)
conv_dim = 10
