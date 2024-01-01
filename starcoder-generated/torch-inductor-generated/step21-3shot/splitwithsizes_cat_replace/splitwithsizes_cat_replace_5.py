
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Sequential(torch.nn.Conv2d(3, 5, 3, 1, 2), torch.nn.Conv2d(5, 8, 3, 1, 3))
        self.split = torch.nn.Sequential(torch.nn.Conv2d(3, 10, 2, 1, 0))
    def forward(self, x):
        concatenated_tensor = torch.cat([x, x-x, x*x], dim=1)
        return (concatenated_tensor, torch.split(x, [1, 1, 1], dim=1))
# Inputs to the model
x1 = torch.randn(1, 3, 5, 5)
