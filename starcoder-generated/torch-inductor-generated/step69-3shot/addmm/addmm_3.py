
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
#        self.inp = torch.empty(0, 0, dtype=torch.float)
    def forward(self, x1):
        v1 = torch.abs(torch.mm(x1, x1.transpose(0, 1)))
        v2 = torch.abs(torch.mm(v1, x1))
        v3 = abs(torch.mm(v2, v2))
        return v3, v1
torch.Size([3, 2])
# Inputs to the model
x1 = torch.randn(3, 2, requires_grad=True)
