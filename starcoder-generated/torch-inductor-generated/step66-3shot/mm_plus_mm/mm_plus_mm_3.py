
class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.input3 = Parameter(torch.Tensor(1, 1024))
        torch.nn.init.orthogonal_(self.input3, 2.2)
    def forward(self, x1):
        return torch.mm(x1, x1) + torch.mm(x1, self.input3.t())
# Inputs to the model
input1 = torch.randn(1000, 1000)
