
class Model(torch.nn.Module):
    def forward(self, input1):
        t1 = torch.mm(input1, input1)
        t2 = t1 + t1
        t3 = torch.mm(input1, input1)
        return torch.mm(t2, t3)
# Inputs to the model
input1 = torch.randn(5, 5)
