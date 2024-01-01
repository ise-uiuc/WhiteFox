
class Model(torch.nn.Module):
    def forward(self, input):
#         t1 = torch.mm(input1, input2)
#         t2 = torch.mm(input3, input4)
#         t3 = torch.mm(input2, input3)
#         t4 = torch.mm(input3, input1)
        t1 = torch.mm(input, input)
        t2 = torch.mm(input, input.T)
#         t3 = t1 + t2
        return torch.mm(input, t1 + t2)
# Inputs to the model
input1 = torch.randn(2, 3)
