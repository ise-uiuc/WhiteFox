
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x, x2, x3, x4):
        x2 = F.relu(x2)
        x3 = F.tanh(x3)
        x4 = F.gelu(x4)
        t1 = torch.stack([x, x2, x3, x4])
        t2 = torch.sum(t1, dim=0)
        return t2.pow(2).sum()
# Inputs to the model
x = torch.randn(1, requires_grad=True)
x2 = torch.randn(1, requires_grad=True)
x3 = torch.randn(1, requires_grad=True)
x4 = torch.randn(1, requires_grad=True)
