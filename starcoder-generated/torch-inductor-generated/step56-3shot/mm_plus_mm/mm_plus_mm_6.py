
class Model(torch.nn.Module):
    def forward(self, input1, input2):
        t1 = torch.mm(input1, input2)
        t2 = t1 + torch.mm(input2, input2)
        return t2
# Inputs to the model
input1 = torch.randn(2, 2)
input2 = torch.randn(2, 2)
