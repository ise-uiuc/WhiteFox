
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(1, 1, 1, stride=1)
    def forward(self, x1):
        v1 = torch.Conv2d.forward(self, input)
        v2 = v1 - input
        v3 = torch.relu(v2)
        return v3
# Inputs to the model
# model = Model() # this is to avoid PyTorch checking the forward function for unused parameters.
x = torch.randn(1, 16, 32, 32)
