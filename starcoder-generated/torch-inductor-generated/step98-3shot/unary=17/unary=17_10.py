
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.ConvTranspose2d(3, 4, (3, 3), padding=(0, 2), stride=(2, 2))
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = nn.functional.relu(v1)
        return v2
# Inputs to the model
x1 = torch.randn(1, 3, 6, 6)
