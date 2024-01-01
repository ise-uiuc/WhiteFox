
class Model(torch.nn.Module):
    def forward(self, input1, input2):
        t1 = torch.mm(input2, input2)
        t2 = torch.mv(input1, input1)
        t3 = t1 + t2
        return t3
# Inputs to the model
input1 = torch.randn(4, 4)
input2 = torch.randn(4, 4)
