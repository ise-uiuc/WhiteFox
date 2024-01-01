
class Model(torch.nn.Module):
    def forward(self, input1):
        t1 = torch.mm(input1, input1)
        t2 = torch.mm(t1, input1)
        t3 = torch.mm(input1, input1)
        return t2 + t3
# Inputs to the model
input1 = torch.randn(16, 64)
