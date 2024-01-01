
class Model(torch.nn.Module):
    def forward(self, input1, input2, input3):
        A = torch.mm(input1, input2)
        B = torch.mm(input1, input2)
        C = torch.mm(input1, input2)
        C = torch.mm(input1, input2)
        C = torch.mm(input1, input2)
        C = torch.mm(input1, input2)
        return (torch.mm(input1, input2) + torch.mm(input1, input2))
# Inputs to the model
input1 = torch.randn(6, 6)
input2 = torch.randn(6, 6)
input3 = torch.randn(6, 6)
