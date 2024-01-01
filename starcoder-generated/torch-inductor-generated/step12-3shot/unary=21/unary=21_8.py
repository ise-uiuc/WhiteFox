
class ModelTanh(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = torch.nn.Sequential(
            torch.nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        )
        self.tanhconv = torch.nn.Conv2d(256, 1, 1)
    def forward(self, x):
        v = self.layer(x)
        v = torch.tanh(v)
        out = self.tanhconv(v)
        return out
# Inputs to the model
