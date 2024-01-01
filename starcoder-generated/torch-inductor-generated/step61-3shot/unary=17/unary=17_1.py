
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = torch.nn.Sequential(torch.nn.ConvTranspose2d(1, 8, 3, stride=2), torch.nn.PReLU())
    def forward(self, x):
        v1 = self.model(x)
        v2 = torch.relu(v1)
        return v2
# Inputs to the model
x = torch.randn(1, 1, 10, 10)
