
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(31, 64, 3, stride=2, padding=1)
    def forward(self, x1, other=None, activation=None):
        v1 = self.conv(x1)
        if activation == None:
            activation = torch.nn.PReLU(1)
        if other == None:
            other = activation(torch.randn(v1.shape).to(x1.device))
        v2 = v1 + other
        return v2
# Inputs to the model
x1 = torch.randn(1, 31, 64, 64).to('cpu')
