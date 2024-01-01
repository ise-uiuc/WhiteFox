
class Model(torch.nn.Module):
    def forward(self, input1, input2, input3):
        t1 = torch.mm(input1, input2)
        t2 = torch.mm(input3, input1)
        t3 = torch.mm(input2, input3)
        s1 = torch.mm(input3, input3)
        return t1 + s1 + t2 + t3 + torch.mm(input2, input2)
# Inputs to the model
input1 = torch.randn(64, 64)
input2 = torch.randn(64, 64)
input3 = torch.randn(64, 64)
