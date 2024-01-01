
class Model(torch.nn.Module):
    def forward(self, input1):
        t1 = torch.mm(input1, input1)
        t2 = torch.mm(input1, input1)
        return torch.mm(t2, t1)
# Inputs to the model
input1 = torch.randn(4, 4)
