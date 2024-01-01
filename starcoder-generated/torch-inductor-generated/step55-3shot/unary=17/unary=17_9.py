
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.convt1 = torch.nn.ConvTranspose2d(1, 1, 1)
        self.convt2 = torch.nn.ConvTranspose2d(1, 1, 1)
    def forward(self, x1):
        v1 = self.convt2(x1)
        v2 = torch.relu(v1)
        v3 = self.convt1(x1)
        v4 = torch.sigmoid(v3)
        return v4
# Inputs to the model
x1 = torch.randn(1, 1, 158, 144)
