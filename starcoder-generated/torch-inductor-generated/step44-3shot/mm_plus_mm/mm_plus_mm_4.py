
class Model(torch.nn.Module):
    def forward(self, input):
        t1 = torch.mm(torch.mm(input, input), torch.mm(input, input))
        t2 = torch.mm(torch.mm(input, input), torch.mm(input, input))
        t3 = torch.mm(torch.mm(input, input), torch.mm(input, input))
        t4 = torch.mm(torch.mm(input, input), torch.mm(input, input))
        t5 = t1 + t2 + t3 + t4
        t6 = torch.mm(input, input)
        t7 = t5 + t6
        t8 = torch.mm(input, input)
        t9 = t6 + t8
        t10 = t7 + t9
        return t10
# Inputs to the model
input1 = torch.randn(2, 2)
