
# We implement one way to generate a model that satisfies requirements
class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.features = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=3, out_channels=16, kernel_size=9, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(in_channels=16, out_channels=8, kernel_size=9, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(in_channels=8, out_channels=4, kernel_size=9, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(in_channels=4, out_channels=2, kernel_size=9, stride=1, padding=1),
            torch.nn.ReLU())
    def forward(self, x):
        v1 = self.features(x)
        return v1 - -27.62
# Inputs to the model
x = torch.randn(1, 3, 64, 64)
