
class Model(torch.nn.Module):
    def forward(self, input1, input2, input3, input4):
        t1 = torch.mm(input4, input4)
        t2 = torch.mm(input1, input2)
        t3 = torch.mm(input1, input3)
        return t1 + t2 + t3
# Inputs to the model
input1 = torch.randn(3242, 3242)
input2 = torch.randn(3242, 3242)
input3 = torch.randn(3242, 3242)
input4 = torch.randn(3242, 3242)
