
class Model(torch.nn.Module):
    def forward(self, input1, input2, input3):
        t1 = torch.mm(input1, input2)
        t2 = torch.mm(t1, input3)
        t4 = torch.mm(input1, input1)
        t3 = t1 + t2 + t4
        return t3
# Inputs to the model
input1 = torch.randn(1, 128)
input2 = torch.randn(768, 128)
input3 = torch.randn(768, 768)
