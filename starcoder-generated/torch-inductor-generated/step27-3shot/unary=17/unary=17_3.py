
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.features = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(1, 32, (3, 3), stride=2),
            torch.nn.Conv2d(32, 64, (3, 3), stride=2),
            torch.nn.ReLU(),
        )
        self.fc = torch.nn.Linear(50176, 10)
    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        output = torch.nn.functional.relu(x)
        return x
# Inputs to the model
x1 = torch.randn(1, 3, 32, 32)
