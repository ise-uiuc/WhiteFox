
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        v1 = torch.nn.functional.relu(torch.nn.functional.conv2d(x, torch.randn(64, 2, 3, 3), groups=2))
        return torch.nn.functional.conv2d(v1, torch.randn(1, 4, 3, 3))
# Input to the model
x = torch.randn(1, 2, 64, 64)
