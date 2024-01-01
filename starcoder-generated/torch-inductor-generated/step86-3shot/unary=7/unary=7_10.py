
class Model(torch.nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.layers = torch.nn.Sequential(
            torch.nn.Linear(1, channels),
            torch.nn.ReLU(),
            torch.nn.Linear(channels, channels)
        )

    def forward(self, x1):
        l1 = self.layers(torch.clamp(x1, min=0))
        l2 = l1 * (torch.clamp(l1 + 3, max=6))
        l3 = l2 / 6
        return x1 * l3

# Initializing the model
m = Model(channels)

# Input to the model
x1 = torch.randn(1, 1)
