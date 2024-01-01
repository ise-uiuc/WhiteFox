
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = torch.nn.Sequential(
            torch.nn.Conv2d(1, 8, 3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(8, 8, 1, stride=1, padding=0)
        )
    def forward(self, x1):
        v1 = self.model(x1)
        return v1
# Inputs to the model
x1 = torch.randn(1, 1, 224, 224)
