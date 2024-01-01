
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.permute = torch.nn.Permutation([1, 2])
    def forward(self, x1, x2):
        v1 = self.permute(x1)
        v2 = self.permute(x2)
        return torch.bmm(v1, v2)
# Inputs to the model
x1 = torch.randn(1, 2, 2)
x2 = torch.randn(1, 2, 2)
