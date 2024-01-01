
class Model(torch.nn.Module):
    def forward(self, input1, input2):
        t1 = torch.mm(input1, input2)
        return t1[0], (t1[1], t1[1])
# Inputs to the model
input1 = torch.randn(5, 5)
input2 = torch.randn(5, 5)
