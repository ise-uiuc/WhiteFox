
class Model(torch.nn.Module):
    def forward(self, input1):
        t1 = torch.mm(input1, input1)
        t2 = t1 + t1
        t3 = t2 - t1
        return t3 - t1
# Inputs to the model
input1 = torch.randn(32, 32)
