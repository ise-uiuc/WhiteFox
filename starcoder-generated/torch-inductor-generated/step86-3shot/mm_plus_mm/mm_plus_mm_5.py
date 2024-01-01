
class Model(torch.nn.Module):
    def forward(self, input1, input2, input11=1):
        t1 = torch.mm(input1, input2)
        t2 = torch.cat([input1, input1 + input11])
        return torch.mm(t1, t2)
# Inputs to the model
input1 = torch.randn(8, 8)
input2 = torch.randn(8, 8)
