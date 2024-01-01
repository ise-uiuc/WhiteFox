
class Model(nn.Module):
    def forward(self, input1):
        y = torch.mm(input1, input1)
        y = y.mm(input1)
        return y
# Inputs to the model
input1 = torch.randn(16, 16)
