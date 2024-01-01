
class Model(torch.nn.Module):
    def forward(self, input1, input2, input3):
        t1 = torch.mm(input1, input3)
        t2 = torch.mm(input2, input3)
        t3 = torch.mm(t1, t2)
        return t3
# Inputs to the model
input1 = torch.randn(4, 4)
input2 = torch.randn(4, 4)
input3 = torch.randn(4, 4)
