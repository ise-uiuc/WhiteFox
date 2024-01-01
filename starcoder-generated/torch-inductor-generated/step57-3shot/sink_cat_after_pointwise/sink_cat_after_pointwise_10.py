
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(1, 3, kernel_size=(3, 5), stride=(3, 5))
    def forward(self, x):
        x = self.conv(x)
        x = torch.cat([x], dim=1)
        x = x.view(x.shape[0], -1)
        x = torch.relu(x)
        return x
# Inputs to the model
x = torch.randn(2, 1, 10, 10)
