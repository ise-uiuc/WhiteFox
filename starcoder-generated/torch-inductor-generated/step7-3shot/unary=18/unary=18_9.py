
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.seq = torch.nn.Sequential(
            torch.nn.Conv2d(112, 256, 5, stride=1, padding=2),
            torch.nn.Conv2d(112, 128, 5, stride=1, padding=2),
            torch.nn.Conv2d(112, 512, 3, stride=1, padding=1),
            torch.nn.Conv2d(112, 256, 5, stride=1, padding=2),
            torch.nn.Conv2d(112, 128, 5, stride=1, padding=2),
            torch.nn.AdaptiveAvgPool2d(output_size=(1, 1)),
            torch.nn.Flatten(),
            torch.nn.Linear(64, 1),
            torch.nn.Sigmoid()
        )
    def forward(self, x1):
        v1 = self.seq(x1)
        return v1
# Inputs to the model
x1 = torch.randn(1, 112, 64, 64)
