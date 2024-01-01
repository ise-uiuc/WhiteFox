
class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv0 = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=2)
    def forward(self, x1):
        v0 = F.relu(self.conv0(x1))
        v1 = v0.permute(0, 2, 3, 1)
        return x1.permute(0, 3, 2, 1)
# Inputs to the model
x1 = torch.randn(1, 1, 10, 10)
