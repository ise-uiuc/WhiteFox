
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.permute = torch.nn.functional.permute
    def forward(self, x1, x2):
        v1 = self.permute(x1, (0,2,1))
        return self.permute(torch.matmul(x2, v1), (0,2,1))
# Inputs to the model
x1 = torch.randn(1, 2, 2)
x2 = torch.randn(1, 2, 2)
