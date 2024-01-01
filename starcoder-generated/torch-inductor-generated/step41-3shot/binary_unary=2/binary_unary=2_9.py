
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(1, 3, 3, stride=1, padding=2)
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = v1 - 0.5
        t0 = torch.tensor([[[[-0.4566]], [[-0.4566]], [[-0.4566]]]]).to("cuda")
        v3 = t0 + v1
        return F.softmax(v3)
# Inputs to the model
x1 = torch.randn(1, 1, 8, 8).to("cuda")
