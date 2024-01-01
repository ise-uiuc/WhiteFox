
class Model(torch.nn.Module):
    def forward(self, input1, input2, input3):
        t1 = torch.mm(input1, input2)
        t2 = t1 * t1
        t3 = t2 - t1
        t4 = torch.mm(t3, input3)
        t5 = t3 + t2
        t6 = torch.mm(t5, input3)
        return (t4 + t6)
# Inputs to the model
input1 = torch.randn(14, 13)
input2 = torch.randn(14, 13)
input3 = torch.randn(13, 28)
