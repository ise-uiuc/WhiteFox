
class Model(torch.nn.Module):
    def forward(self, input1, input2):
        t1 = torch.mul(input1, input2)
        t2 = torch.mul(input2, input1)
        return t1 + t2
# Inputs to the model
input1 = torch.randn(6, 6)
input2 = torch.randn(6, 6)
