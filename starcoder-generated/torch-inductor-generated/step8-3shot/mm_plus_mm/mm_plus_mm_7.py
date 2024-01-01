
class Model(torch.nn.Module):
    def forward(self, input1):
        t1 = torch.mm(input1, input1)
        return t1
# Inputs to the model
input1 = torch.randn(5, 5)
