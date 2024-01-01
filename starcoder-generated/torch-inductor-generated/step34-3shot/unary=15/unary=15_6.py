
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(1, 1, 3, stride=1, padding=1)
        self.linear = nn.Linear(1,1)
    def forward(self, x):
        x = self.conv(x)
        x = x.squeeze()
        x = self.linear(x)
        return x.view(-1)
# Inputs to the model
# Only one channel with 98 x 98 is needed to trigger the error.
x1 = torch.arange(1, 99).view(1, 1, 39, 39)
