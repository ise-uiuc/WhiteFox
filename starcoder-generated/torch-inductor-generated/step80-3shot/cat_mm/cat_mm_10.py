
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1, x2):
        v1 = torch.cat([torch.Tensor([10]), torch.rand(1)], 0)
        v2 = torch.cat([torch.Tensor([10.]), torch.rand(1)], 0)
        v3 = torch.cat([torch.Tensor([1.]), torch.rand(2)], 0)
        v4 = torch.cat([torch.Tensor([1.]), torch.rand(2)], 0)
        v5 = torch.cat([torch.rand(2), torch.Tensor([1.])], 0)
        v6 = torch.cat([torch.rand(3), torch.Tensor([1.])], 0)
        x = torch.cat([v1, v2, v3, v4, v5, v6], 0)
        return torch.mm(x, x.t())
# Inputs to the model
x1 = torch.ones([10], dtype=torch.float)
x2 = torch.rand([10])
