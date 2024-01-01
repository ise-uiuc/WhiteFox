
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.ConvTranspose2d(1, 9, 2, stride=1, bias=True)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = F.relu(v1)
        return v2
# Inputs to the model
x = torch.randn(1, 1, 7, 7)
