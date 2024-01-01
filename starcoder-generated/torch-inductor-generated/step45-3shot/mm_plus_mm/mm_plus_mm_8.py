
class Model(torch.nn.Module):
    def forward(self, i1, i2, i3, i4):
        t1 = torch.bmm(i1, i2)
        t2 = torch.einsum('nc, mnc -> nm', i3, i4)
        t3 = t2 / t2.sum()
        return t1 + t3
# Inputs to the model
i1 = torch.randn(6, 3, 5)
i2 = torch.randn(6, 5, 7)
i3 = torch.randn(3, 7)
i4 = torch.randn(6, 7, 5)
