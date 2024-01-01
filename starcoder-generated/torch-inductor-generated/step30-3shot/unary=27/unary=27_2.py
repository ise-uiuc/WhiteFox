
class Model(torch.nn.Module):
    def __init__(self, min, max, hidden):
        super().__init__()
        self.conv = torch.nn.Conv2d(8, 5, 1, stride=2, padding=1)
        self.min = min
        self.max = max
        self.model1 = torch.nn.Sequential(
            torch.nn.Linear(5, hidden),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden, 2),
            torch.nn.ReLU()
        )
        self.model2 = torch.nn.Sequential(
            torch.nn.Linear(2, hidden),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden, 5),
            torch.nn.ReLU()
        )
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = torch.clamp_min(v1, self.min)
        v3 = torch.clamp_max(v2, self.max)
        v4 = self.model1(v3)
        v5 = self.model2(v4)
        return v5
min = 0.6
max = 0.8
hidden = 120
# Inputs to the model
x1 = torch.randn(1, 8, 200, 250)
