
class Model(torch.nn.Module):
    def forward(self, input1):
        v1 = torch.mm(input1, input1)
        v2 = torch.mm(input1, input1)
        v3 = torch.mm(input1, input1)
        return v1 + v2 + v3
# Inputs to the model
input1 = torch.randn(1, 1)
