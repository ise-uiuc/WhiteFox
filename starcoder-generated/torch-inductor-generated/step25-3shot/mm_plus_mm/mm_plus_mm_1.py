
class Model(torch.nn.Module):
    def forward(self, input1, input2, input3):
        t1 = torch.einsum("ij,jk->ik", (input1, input2))
        t2 = torch.einsum("ij,jk->ik", (t1, input3))
        return torch.mm(t2, input1)
# Inputs to the model
input1 = torch.randn(3, 6)
input2 = torch.randn(6, 3)
input3 = torch.randn(6, 3)
