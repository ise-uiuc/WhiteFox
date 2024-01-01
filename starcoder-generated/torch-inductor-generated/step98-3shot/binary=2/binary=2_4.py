
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv1d(512, 120, 1)
    def forward(self, x):
        v1 = self.conv(x)
        v2 = v1 - 2468.488972862732
        return v2
# Inputs to the model
x = torch.randn(1, 512, 43179)
