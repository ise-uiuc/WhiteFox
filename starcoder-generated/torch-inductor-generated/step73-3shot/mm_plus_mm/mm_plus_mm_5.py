
class Model(torch.nn.Module):
    def forward(self, input1, input2, input3):
        return torch.mm(input1, input1) + torch.mm(input2, input2) + torch.mm(input3, input3)
# Inputs to the model
input1 = torch.randn(969, 969)
input2 = torch.randn(969, 969)
input3 = torch.randn(969, 969)
