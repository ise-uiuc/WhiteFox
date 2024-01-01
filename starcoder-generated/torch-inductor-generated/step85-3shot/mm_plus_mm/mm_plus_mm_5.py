
class Model(torch.nn.Module):
    def forward(self, input1, input2):
        t1 = torch.mm(input1, input1)
        t2 = torch.mm(input1, input2)
        t3 = torch.mm(input2, input1)
        t4 = torch.mm(input2, input2)
        t5 = t1 * t3 + t2 + t4
        return t5
# Inputs to the model
input1 = torch.randn(1000, 1000)
input2 = torch.randn(897, 1255)
