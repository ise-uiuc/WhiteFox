
class Model(torch.nn.Module):
    def forward(self, input1, input2):
        t1 = torch.mm(input1, input2)
        return t1
# Inputs to the model
input1 = torch.randn(100, 100)
input2 = torch.randn(100, 100)
