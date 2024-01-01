
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.constant1 = torch.randn(1)
        self.constant2 = torch.randn(1)
        self.constant3 = torch.randn(1)
        self.constant4 = torch.randn(1)
        self.constant5 = torch.randn(1)
        self.constant6 = torch.randn(1)
        self.constant7 = torch.randn(1)
    def forward(self, x1):
        v3 = self.constant7 * 2
        v1 = self.constant2 + x1
        v2 = self.constant1 * v1
        return v2
# Inputs to the model
x1 = torch.randn(1, device='cpu')
