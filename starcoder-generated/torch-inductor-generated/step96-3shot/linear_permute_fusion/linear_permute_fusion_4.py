
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = torch.nn.Linear(2, 2)
        self.linear2 = torch.nn.Linear(2, 2)
    def forward(self, x1):
        a1 = x1
        v1 = torch.nn.functional.linear(x1, self.linear1.weight, self.linear1.bias)
        v2 = torch.nn.functional.linear(v1, self.linear2.weight, self.linear2.bias)
        a2 = v2.permute(0, 2, 1)
        v3 = pytorch_internal.prim.NumToTensor(1.)
        v4 = v2 - v3
        return (v2, a1, a2, v4)
# Inputs to the model
x1 = torch.randn(1, 2, 2, device='cpu', requires_grad=True)
