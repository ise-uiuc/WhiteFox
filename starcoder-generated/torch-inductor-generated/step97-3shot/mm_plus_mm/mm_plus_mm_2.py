
class Model(torch.nn.Module):
    def forward(self, input1, input2, input3, input4):
        mm = torch.mm(input1, input2)
        mm2 = torch.mm(input2, input3)
        return mm + mm2
# Inputs to the model
input1 = torch.randn(5, 5)
input2 = torch.randn(5, 5)
input3 = torch.randn(5, 5)
input4 = torch.randn(5, 5)
