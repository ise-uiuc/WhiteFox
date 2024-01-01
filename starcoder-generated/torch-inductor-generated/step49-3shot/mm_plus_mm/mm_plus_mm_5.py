
class Model(torch.nn.Module):
    def forward(self, input1, input2):
        input2 = input2 - input1
        return torch.mm(input1, input2)
# Inputs to the model
input1 = torch.randn(55, 55)
input2 = torch.randn(55, 55)
