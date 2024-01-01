
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_t = torch.nn.ConvTranspose2d(3, 12, 1, stride=1, padding=0, bias=False)
        self.fc = torch.nn.Linear(12, 1)
    def forward(self, x):
        x1 = self.conv_t(x)
        x2 = self.fc(x1.flatten(1))
        return x2
# Inputs to the model
x = torch.randn(2, 3, 10, 20)
