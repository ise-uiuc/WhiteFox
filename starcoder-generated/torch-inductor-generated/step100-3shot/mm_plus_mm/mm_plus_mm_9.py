
class Model(torch.nn.Module):
    def forward(self, i1, i2, i3, i4):
        t1 = torch.mm(i1, i2)
        t2 = torch.mm(i3, i4)
        return torch.matmul(t1, t2)
# Inputs to the model
i1 = torch.randn(3, 3)
i2 = torch.randn(3, 3)
i3 = torch.randn(3, 3)
i4 = torch.randn(3, 3)
