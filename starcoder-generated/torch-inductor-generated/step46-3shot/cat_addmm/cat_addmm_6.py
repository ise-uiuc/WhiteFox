
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.stack = torch.stack
    def forward(self, x):
        x = torch.stack((x, x), dim=0)
        x = x.view([2, -1])
        return x
# Inputs to the model
x = torch.randn(2)
y = torch.randn(4)
