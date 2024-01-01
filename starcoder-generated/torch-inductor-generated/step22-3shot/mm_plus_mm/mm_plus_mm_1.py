
class Model(torch.nn.Module):
    def forward(self, input1, input2, input3):
        mm1 = torch.mm(input1, input2)
        mm2 = torch.mm(mm1, input2)
        return mm2.mm(input3)
# Inputs to the model
input1 = torch.randn(10, 10)
input2 = torch.randn(10, 10)
input3 = torch.randn(10, 10)
