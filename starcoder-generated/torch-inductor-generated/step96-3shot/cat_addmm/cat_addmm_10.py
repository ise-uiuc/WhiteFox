
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv1d(4, 6, kernel_size=1)
        self.conv2 = nn.Conv2d(4, 6, stride=2, kernel_size=3)
    def forward(self, x):
        x = self.conv1(x)
        x = x.flatten(start_dim=1)
        x = self.conv2(x)
        return x
# Inputs to the model
x = torch.randn(1, 4, 10, 10)
