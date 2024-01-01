
class Model(torch.nn.Module):
    def forward(self, input1, input2, input3, input4, input5, input6):
        t1 = torch.mm(input3, input4)
        t2 = torch.mm(input1, input2)
        t3 = input1.mm(input4)
        t4 = torch.mm(t1, input2)
        t5 = torch.mm(input5, input6)
        t6 = torch.mm(input1, input4)
        t7 = t5 + t3 + torch.mm(t6, input6) - input2.mm(input4)
        t8 = t2 + t7
        t9 = torch.mm(input4, input2)
        t10 = t8.mm(t9)
        return torch.mm(t8, t10)
