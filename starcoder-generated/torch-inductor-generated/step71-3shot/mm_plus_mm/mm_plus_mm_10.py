
class Model(nn.Module):
    def forward(self, input1, input2, input3, input4, input5, input6, input7, input8):
        t1 = torch.mm(input1, input1.transpose(1, 0))
        t2 = torch.mm(input2, input2.transpose(1, 0))
        t3 = torch.mm(input3, input3.transpose(1, 0))
        t4 = torch.mm(input4, input4.transpose(1, 0))
        return t1 + t2 + t3 + t4
# Inputs to the model
input1 = torch.randn(2, 9)
input2 = torch.randn(2, 9)
input3 = torch.randn(2, 9)
input4 = torch.randn(2, 9)
