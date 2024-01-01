
class Model(torch.nn.Module):
    def forward(self, *inputs):
        t1 = torch.mm(inputs[0], inputs[1])
        t2 = torch.mm(inputs[2], inputs[3])
        t3 = torch.mm(inputs[4], inputs[5])
        t4 = torch.mm(inputs[6], inputs[7])
        t5 = torch.mm(inputs[8], inputs[9])
        t6 = torch.mm(inputs[10], inputs[11])
        t7 = torch.mm(inputs[12], inputs[13])
        t8 = t1 + t2 + t3 + t4 + t5 + t6 + t7
        return t8
# Inputs to the model
input1 = torch.randn(4, 4)
input2 = torch.randn(4, 4)
input3 = torch.randn(4, 4)
input4 = torch.randn(4, 4)
input5 = torch.randn(4, 4)
input6 = torch.randn(4, 4)
input7 = torch.randn(4, 4)
input8 = torch.randn(4, 4)
input9 = torch.randn(4, 4)
input10 = torch.randn(4, 4)
input11 = torch.randn(4, 4)
input12 = torch.randn(4, 4)
input13 = torch.randn(4, 4)
