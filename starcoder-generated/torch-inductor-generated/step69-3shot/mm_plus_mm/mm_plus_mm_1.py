
class Model(torch.nn.Module):
    def forward(self, input1, input2, input3):
        v1 = torch.mm(input1, input2)
        v2 = torch.mm(input1, input2)
        input3 = torch.mm(input1, input2)
# Inputs to the model
input3 = torch.randn(8, 8)
input2 = torch.randn(8, 8)
input1 = torch.randn(8, 8)
