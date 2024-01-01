
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.seq = torch.nn.Sequential(
            torch.nn.Conv2d(28, 64, kernel_size=(3, 3), stride=1, padding=(1, 1)),
            torch.nn.MaxPool2d(2, stride=2, padding=0),
            torch.nn.Conv2d(64, 128, kernel_size=(3, 3), stride=1, padding=(1, 1)),
            torch.nn.MaxPool2d(2, stride=2, padding=0),
            torch.nn.Conv2d(128, 64, kernel_size=(3, 3), stride=1, padding=(1, 1)),
            torch.nn.MaxPool2d(2, stride=2, padding=0),
            torch.nn.Flatten(),
            torch.nn.Linear(576, 10)
        )
    def forward(self, x1):
        v1 = self.seq(x1)
        v2 = torch.sigmoid(v1)
        return v2
# Inputs to the model
x1 = torch.randn(1, 28, 32, 32)
