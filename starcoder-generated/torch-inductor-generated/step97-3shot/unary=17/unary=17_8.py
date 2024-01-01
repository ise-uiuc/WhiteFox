
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.convlayer1 = torch.nn.ConvTranspose2d(1, 1, (1, 3), stride=2, padding=(0, 1), bias=False)
        self.convlayer2 = torch.nn.ConvTranspose2d(1, 1, (1, 3), stride=2, padding=(0, 1), bias=False)
    def forward(self, x1):
        v1 = self.convlayer1(x1)
        v2 = torch.relu(v1)
        v3 = torch.sigmoid(v2)
        v4 = self.convlayer2(x1)
        v5 = torch.relu(v4)
        v6 = torch.sigmoid(v5)
        return v3, v6
# Inputs to the model
x1 = torch.randn(1, 1, 36, 36)
