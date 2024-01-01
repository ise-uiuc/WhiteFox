
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_block = torch.nn.Sequential(
            torch.nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2, 2),

            torch.nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2, 3),

            torch.nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2, 2),

            torch.nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2, 2),
        )
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(128 * 38 * 38, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 2)
        )
    def forward(self, x1):
        v1 = self.conv_block(x1)
        v2 = v1.reshape(v1.size(0), -1)
        v3 = self.fc(v2)
        return v3
# Inputs to the model
x1 = torch.randn(1, 3, 300, 300)
