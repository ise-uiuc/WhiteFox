
class Model(torch.nn.Module):
    def forward(self, input1):
        mm1 = input1.mm(input1)
        mm2 = input1.mm(input1)
        return mm1 + mm2
# Inputs to the model
input1 = torch.randn(2, 2)
