
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = torch.nn.Sequential(torch.nn.Conv2d(2, 3, 3, bias=True), torch.nn.ReLU())
    def forward(self, d):
        a2 = self.layer(d)
        return a2
# Inputs to the model
d = torch.randn(1, 2, 3, 3)
