
class Model(torch.nn.Module):
    def forward(self, input1, input2, input3):
        if len(input1.shape) == 2:
            t1 = torch.mm(input1, input2)
            t2 = torch.mm(input3, input3)
        else:
            t1 = torch.matmul(input1, input2)
            t2 = torch.matmul(input3, input3)
        return torch.mul(t2, t1).sum() * 100.
# Inputs to the model
input1 = torch.randn(5, 5)
input2 = torch.randn(5, 5)
input3 = torch.randn(5, 5)
