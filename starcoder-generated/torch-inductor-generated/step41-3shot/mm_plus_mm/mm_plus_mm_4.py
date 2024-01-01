
class Model(torch.nn.Module):
    def forward(self, input):
        t1 = torch.mm(input, input)
        t2 = torch.mm(input, input)
        t3 = torch.mm(input, input)
        t4 = t1 + t2 + t3
        t5 = torch.mm(input, input)
        t6 = t4 + t5
        t7 = torch.mm(input, input)
        t8 = torch.mm(input, input)
        return t7 + t3 + t4 + t8 + t6
# Inputs to the model
input = torch.randn(5, 5)
