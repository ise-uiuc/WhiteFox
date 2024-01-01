
class ModelTanh(torch.nn.Module):
    def __init__(self):
        super(ModelTanh, self).__init__()
        self.f = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=1, stride=1, padding=0),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32, 3, kernel_size=1, stride=1, padding=0),
            nn.Tanh(),
        )

    def forward(self, x):
        return self.f(x)
# Inputs to the model
x1 = torch.randn(1, 32, 10, 10)
