
class Model(torch.nn.Module):
    def __init__(self):
    super().__init__()
    self.layer = torch.nn.Sequential(torch.nn.Conv2d(2, 20, 5, stride=1, padding=0), torch.nn.LeakyReLU(), torch.nn.Conv2d(20, 50, 5, stride=1, padding=0), torch.nn.LeakyReLU(), torch.nn.Conv2d(50, 10, 5, stride=1, padding=0), torch.nn.MaxPool2d(kernel_size=2, stride=1), torch.nn.Flatten(), torch.nn.Linear(4, 1), torch.nn.Sigmoid())
    def forward(self, x1):
        v1 = self.layer(x1)
        return v1
# Inputs to the model
x1 = torch.randn(1, 2, 28, 28)
