
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.v1 = torch.Tensor(2, 3)
        self.v1.uniform_(-10, 10)
    def forward(self, x1, x2):
        return torch.bmm(self.v1, x1)
# Inputs to the model
x1 = torch.randn(1, 2, 3)
x2 = torch.randn(1, 2, 3)
