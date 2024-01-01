
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.model = nn.Sequential(nn.Conv2d(2, 3, kernel_size=3), nn.ReLU(inplace=True), nn.Conv2d(3, 4, kernel_size=5))
    def forward(self, x):
        return self.model(x)
# Inputs to the model
x = torch.randn(2, 2, 5, 6)
