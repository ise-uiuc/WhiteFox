
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv1d(3, 8, 3, stride=1, padding=1)

    def forward(self, x):
        v1 = self.conv(x)
        v2 = torch.cumsum(v1, 1)
        return v2
        
# Initializing the model
m = Model()

# Inputs to the model
x = torch.randn(1, 3, 64)
