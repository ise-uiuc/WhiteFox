
class Model(torch.nn.Module):
    def forward(self, input1, input2):
        t1 = torch.mm(input2, input1)
        return t1 + t1
# Inputs to the model
input1 = torch.randn(2, 2)
input2 = torch.randn(2, 2)
