
class Model(torch.nn.Module):
    def __init__(self, v_size):
        super().__init__()
        self.v_size = v_size
    def forward(self, x1):
        v1, v2 = self.helper(x1)
        return torch.cat([v1, v2, v1, v2, v1, v2, v1, v2, v1, v2, v1, v2, v1, v2, v1, v2], dim=1)
    def helper(self, x1):
        v1 = torch.mm(x1, x1)
        v2 = torch.mm(x1, x1)
        return v1, v2
# Inputs to the model
x1 = torch.randn(1, 15)
