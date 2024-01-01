
class Model(torch.nn.Module):
    def forward(self, i1, i2, i3, i4, i5, i6, i7, i8, i9):
        h1 = torch.mm(i1, i2)
        h2 = torch.nn.functional.relu(torch.mm(i2, i1))
        h3 = torch.nn.functional.relu(torch.mm(i3, i2))
        h4 = torch.mm(i4, i5)
        h5 = torch.nn.functional.relu(torch.mm(i5, i4))
        h6 = torch.mm(i5, i6)
        h7 = torch.nn.functional.relu(torch.mm(i4, i7))
        h8 = torch.nn.functional.relu(torch.mm(i8, i9))
        return h1 + h2 + h3 + h4 + h5 + h6 + h7 + h8
# Inputs to the model
i1 = torch.randn(8, 1)
i2 = torch.randn(1, 8)
i3 = torch.randn(8, 1)
i4 = torch.randn(1, 8)
i5 = torch.randn(8, 1)
i6 = torch.randn(1, 8)
i7 = torch.randn(8, 1)
i8 = torch.randn(1, 8)
i9 = torch.randn(8, 1)
