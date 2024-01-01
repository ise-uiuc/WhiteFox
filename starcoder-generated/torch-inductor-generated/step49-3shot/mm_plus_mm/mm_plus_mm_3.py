
class Model(torch.nn.Module):
    def forward(self, input1, input2):
        mm1 = torch.mm(input1, input2)
        mm2 = torch.mm(input2, torch.mm(input1, input2))
        return torch.mm(mm1, torch.mm(mm2, input1))
# Inputs to the model
input1 = torch.randn(55, 55)
input2 = torch.randn(55, 55)
