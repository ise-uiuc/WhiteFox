
class Model(torch.nn.Module):
    def forward(self, input1, input2, input3, input4):
        t0 = torch.mm(input1, input3)
        t1 = torch.mm(input1, input1)
        out = torch.mm(input3, input4)
        out = t1 + t0 + out
        return out
# Inputs to the model
input1 = torch.randn(5, 5)
input2 = torch.randn(5, 5)
input3 = torch.randn(5, 5)
input4 = torch.randn(5, 5)
