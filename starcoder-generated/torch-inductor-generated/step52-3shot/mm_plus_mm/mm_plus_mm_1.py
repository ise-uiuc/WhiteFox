
class Model(torch.nn.Module):
    def forward(self, input1):
        i0 = torch.mm(input1, input1)
        return i0
# Inputs to the model
input1 = torch.randn(10, 10)
