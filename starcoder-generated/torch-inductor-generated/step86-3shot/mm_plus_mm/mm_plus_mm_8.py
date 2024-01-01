
class Model(nn.Module):
    def forward(self, input1):
        t1 = torch.mm(input1, input1)
        t2 = torch.mm(input1, input1)
        return t1 + t2
# Inputs to the model
input1 = torch.randn(100, 100)
