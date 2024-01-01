
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.mm1 = torch.nn.Linear(3, 3)
    def forward(self, x1, x2):
        self.mm1.weight = torch.Tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        v1 = torch.mm(x1, x2)
        v2 = v1 + self.mm1(x1)
        return v2
# Inputs to the model
x1 = torch.randn(3, 3)
x2 = torch.randn(3, 3)
