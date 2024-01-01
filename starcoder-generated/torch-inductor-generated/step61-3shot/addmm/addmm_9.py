
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.Tensor(200, 200))
        self.weight.data.normal_()
    def forward(self, x1):
        return torch.mm(x1, torch.mm(x1, x1))
# Inputs to the model
x = torch.randn(200, 200, requires_grad=True)
