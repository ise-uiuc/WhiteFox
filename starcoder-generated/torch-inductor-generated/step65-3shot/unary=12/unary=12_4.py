
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.x1 = torch.nn.Parameter(torch.randn((3)))
    def forward(self):
        v1 = torch.sigmoid(self.x1)
        v2 = v1.mul(self.x1)
        v3 = torch.mul(self.x1, v2)
        return v3
# Inputs to the model
