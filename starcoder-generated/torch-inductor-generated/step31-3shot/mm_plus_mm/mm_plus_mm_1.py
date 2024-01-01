
class Model(torch.nn.Module):
    def forward(self, input1):
        t1 = torch.mm(input1, input1)
        t2 = torch.mm(input1.transpose(2,1), input1)
        return t1 + t2
# Inputs to the model
input1 = torch.randn(216, 27, 10)
