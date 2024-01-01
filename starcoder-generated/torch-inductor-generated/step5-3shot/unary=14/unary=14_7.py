
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.convtranspose = torch.nn.ConvTranspose2d(1, 1, kernel_size=3, stride=1, padding=1, output_padding=1, bias=False)
        self.fc1 = torch.nn.Linear(8, 16)
    def forward(self, x1):
        v1 = self.convtranspose(x1)
        v2 = torch.sigmoid(v1)
        v3 = v1 * v2
        v4 = self.fc1(v3)
        v5 = torch.sigmoid(v3)
        v6 = v4 * v5
        return x1
# Inputs to the model
x1 = torch.randn(1, 1, 8, 8)
