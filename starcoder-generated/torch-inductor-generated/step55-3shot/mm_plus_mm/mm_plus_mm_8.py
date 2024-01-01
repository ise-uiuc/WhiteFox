
class Model(torch.nn.Module):
    def forward(self, input1):
        t1 = torch.mm(input1, input1)
        t2 = torch.mm(input1, input1)
        t3 = t1 + t2
        t4 = torch.mm(input1, input1)
        return t3 + t4
# Inputs to the model
input1 = torch.randn(16, 16)
