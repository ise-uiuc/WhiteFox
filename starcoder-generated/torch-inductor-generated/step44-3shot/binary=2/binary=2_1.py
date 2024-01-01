
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.ConvTranspose1d(2, 5, 2, stride=1, padding=0, output_padding=1)
        self.fc = torch.nn.Linear(5, 7)
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = v1 - 0.154
        v3 = self.fc(v2)
        return v3
# Inputs to the model
x1 = torch.randn(1, 2, 3)
