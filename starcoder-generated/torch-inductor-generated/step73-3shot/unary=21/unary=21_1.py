
class ModelSigmoid(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 32, kernel_size=1, padding=1)
        self.fc = torch.nn.Linear(32 * 8 * 8, 1)
    def forward(self, x):
        x = self.fc(x)
        x = torch.sigmoid(x)
        return x
# Inputs to the model
x = torch.randn(1, 3, 16*56*56)
