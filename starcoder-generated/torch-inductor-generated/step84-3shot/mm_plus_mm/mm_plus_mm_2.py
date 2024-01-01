
class Model(torch.nn.Module):
    def forward(self, a1, b1, c1, d1):
        x = torch.mm(a1, d1)
        y = torch.mm(x, b1)
        x = torch.mm(x, a1)
        z = torch.mm(x, b1)
        return y + z
# Inputs to the model
a1 = torch.Tensor([[1.0]])
b1 = torch.Tensor([[1.0]])
c1 = torch.zeros((1, 1)).float()
d1 = torch.zeros((2, 2)).float()
