
class Model(torch.nn.Module):
    def forward(self, input1, input2, input3, input4):
        t1 = input1 @ input2 + input3 @ input4 + input3 @ input2 + input1 @ input4
        return t1
# Inputs to the model
input1 = torch.randn(6, 8, requires_grad=True)
input2 = torch.randn(8, 6, requires_grad=True)
input3 = torch.randn(8, 128, requires_grad=True)
input4 = torch.randn(128, 6, requires_grad=True)
