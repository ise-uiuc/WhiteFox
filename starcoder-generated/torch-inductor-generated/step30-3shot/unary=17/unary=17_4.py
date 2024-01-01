
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = torch.nn.Conv2d(1024, 512, (1, 1), stride=(1, 1), bias=False)
    def forward(self, x1):
        v1 = self.layer(x1) # Change stride to (2, 2)
        v2 = torch.relu6(v1)
        v3 = v2.view((-1, 512, 2, 2))
        return v3
# Inputs to the model
x1 = torch.randn(1, 1024, 14, 14) # Change 1 to 2
