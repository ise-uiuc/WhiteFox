
class Model(torch.nn.Module):
    def forward(self, input1, input2):
        t1 = torch.mm(input1, input2)
        t2 = torch.mm(input2, input2)
        t3 = t2 + torch.mm(input1, input1)
        return t1 + t3
# Inputs to the model
input1 = torch.randn(3, 3)
input2 = torch.randn(3, 3)
