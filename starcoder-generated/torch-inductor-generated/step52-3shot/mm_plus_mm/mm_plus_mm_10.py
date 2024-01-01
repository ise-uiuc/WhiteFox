
class Model(torch.nn.Module):
    def forward(self, input1):
        t1 = torch.mm(input1, input1)
        t2 = torch.mm(t1, input1)
        t3 = input1 + t1
        return torch.mm(t2, t3)
# Inputs to the model
input1 = torch.randn(11, 11)
