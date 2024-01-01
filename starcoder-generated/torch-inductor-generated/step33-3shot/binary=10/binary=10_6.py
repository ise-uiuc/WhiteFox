
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(128, 128)
        self.conv = torch.nn.Conv1d(3, 128, 1, stride=2, padding=0)
 
    def forward(self, x1, x2):
        v1 = self.linear(x1)
        v2 = self.conv(x2)
        v3 = v1 + v2
        return v3

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 128)
x2 = torch.randn(1, 3, 32)
