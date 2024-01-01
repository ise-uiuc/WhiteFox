
class Model(torch.nn.Module):
    def forward(self, input1, input2, input3, input4):
        input1 = input1.cuda()
        input2 = input2.cuda()
        input3 = input3.cuda()
        input4 = input4.cuda()
        mm1 = torch.mm(input1, input2)
        mm2 = torch.mm(input2, input2)
        mm3 = torch.mm(input1, input3)
        mm4 = torch.mm(input4, input4)
        return mm1 + mm2 + mm3 + mm4
# Inputs to the model
input1 = torch.randn(nRows, nCols)
input2 = torch.randn(nRows, nCols)
input3 = torch.randn(nRows, nCols)
input4 = torch.randn(nRows, nCols)
