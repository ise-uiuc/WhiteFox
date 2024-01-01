


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.convtranspos = torch.nn.ConvTranspose2d(16, 16, 3, padding=1)
        self.convtranspos1 = torch.nn.ConvTranspose2d(16, 16, 3, padding=1)
    def forward(self, x1):
        v1 = torch.nn.functional.relu(x1)
        v2 = self.convtranspos(v1)
        v3 = torch.nn.functional.relu(v2)
        v4 = v3.transpose(2, 1)
        v5 = self.convtranspos1(v4)
        v6 = torch.nn.functional.sigmoid(v5)
        return v6
# Inputs to the model
x1 = torch.randn(1, 16, 224, 224)
