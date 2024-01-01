
class Model(torch.nn.Module):
    def forward(self, input1, input2, input3, input4):
        t1 = torch.einsum('ij, j', [input1, input2])
        t2 = torch.mm(input3, input4)
        t3 = torch.einsum('ij, i, j', [input2, input1, input4])
        t4 = torch.einsum('i, j', [input1, input3])
        return t1 + t2 + t3 + t4
# Inputs to the model
input1 = torch.randn(5, 5)
input2 = torch.randn(5, 5)
input3 = torch.randn(5, 5)
input4 = torch.randn(5, 5)
