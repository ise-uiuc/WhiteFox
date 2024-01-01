
class Model(torch.nn.Module):
    def forward(self, input1, input2, input3, input4):
        t1 = input1 + 2 * input2
        t2 = input3 + 2 * input4
        return t1.mm(t2) + torch.mm(t1, t2)
# Inputs to the model
input1 = torch.randn(3, 3)
input2 = torch.randn(3, 3)
input3 = torch.randn(3, 3)
input4 = torch.randn(3, 3)
