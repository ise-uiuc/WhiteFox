
class Model(torch.nn.Module):
    def __init__(self, in_channels=3, out_channels=32):
        super().__init__()
        self.fc = torch.nn.Linear(in_channels, out_channels)
 
    def forward(self, l1):
        v1 = self.fc(l1)
        v2 = v1 * torch.clamp(torch.min(v1 + 3), min=0, max=6)
        v3 = v2 / 6
        return v3

# Initializing the model
m = Model()

# Inputs to the model
l1 = torch.randn(1, 10, 5)
