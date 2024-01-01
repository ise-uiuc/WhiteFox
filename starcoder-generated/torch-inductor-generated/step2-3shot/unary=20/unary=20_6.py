
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.ConvTranspose3d(1, 1, (1, 3, 3), stride=(1, 2, 2))
    def forward(self, x):
        return F.relu(self.conv1(x))
# Inputs to the model
x = torch.randn(2, 1, 2, 4, 4)
