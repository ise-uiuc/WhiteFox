
class Model(torch.nn.Module):
    def forward(self, input1, input2, input3, input4, input5, input6):
        t0 = torch.mm(input5, input3)
        a1 = input1
        y1 = t0
        a2 = input1
        y2 = torch.mm(input5, a2)
        a3 = input1
        y3 = torch.mm(input5, a3)
        a4 = input6
        y4 = torch.mm(input5, a4)
        t1 = y4 + y2 + y3
        t2 = torch.mm(input4, torch.mm(input3, torch.mm(input2, torch.mm(input1, t1))))
        return t2
# Inputs to the model
input1 = torch.randn(1, 1, 5)
input2 = torch.randn(1, 1, 5)
input3 = torch.randn(1, 1, 5)
input4 = torch.randn(1, 1, 5)
input5 = torch.randn(1, 1, 5)
input6 = torch.randn(1, 1, 5)
