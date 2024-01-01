
class Model(torch.nn.Module):
    def forward(self, input):
        t1 = torch.mm(input, input)
        t2 = torch.mm(input, input)
        t3 = t1 + t2
        t4 = torch.mm(t3, t3)
        return t3 - t4
# Inputs to the model
input = torch.randn(2, 2)
