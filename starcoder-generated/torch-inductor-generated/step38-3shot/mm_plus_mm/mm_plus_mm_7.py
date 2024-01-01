
class Model(torch.nn.Module):
    def forward(self, input1, input2):
        t1 = torch.mm(input1, input2)
        t2 = torch.mm(input2, input1)
        return t1 + t2
# Inputs to the model
input = torch.randn(3,10)
