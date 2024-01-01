
class Model(torch.nn.Module):
    def forward(self, input1, input2):
        t0 = torch.mm(input2, input1)
        t1 = torch.mm(input1, input1)
        out = t1 + t0
        return out
# Inputs to the model
input1 = torch.randn(5, 5)
input2 = torch.randn(5, 5)
