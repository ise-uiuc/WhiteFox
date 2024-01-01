
class Model(torch.nn.Module):
    def forward(self, input1, input2):
        t1 = torch.mm(input1, input1)
        t3 = torch.mm(input2, input2)
        t4 = t1 + t3
        return t4
# Inputs to the model
input1 = torch.randn(298, 298)
input2 = torch.randn(298, 298)
