
class Model(torch.nn.Module):
    def __init__(self, x2):
        super().__init__()
        self.x2 = x2
    def forward(self, x1):
        v1 = self.x2
        v2 = torch.relu(v1)
        return v2
# Inputs to the model
x1 = torch.randn(1, device='cpu')
x2 = torch.eye(2, device='cpu')
