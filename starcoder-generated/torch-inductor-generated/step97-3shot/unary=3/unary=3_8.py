
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.v1 = torch.nn.Parameter(torch.empty(512, 512, dtype=torch.float64, device='cuda'))
        self.v1.uniform_(1.0, 10.0)
        self.v2 = torch.nn.Parameter(torch.empty(512, 512, dtype=torch.float64, device='cuda'))
        self.v2.uniform_(1.0, 10.0)
        self.conv2 = torch.nn.Conv2d(3, 64, 3, stride=1, padding=1)
    def forward(self, x1):
        v1 = (x1 * self.v2) + self.v1
        return self.conv2(v1)
# Inputs to the model
x1 = torch.randn(1, 3, 224, 224, device='cuda')
