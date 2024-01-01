
class Model(torch.nn.Module):
    def forward(self, input1, input2, input3, input4):
        x = input1 * torch.mm(input2, input3)
        y = input1 * torch.mm(input4, input4)
        z = torch.mm(x, y)
        return z
# Inputs to the model
input1 = torch.randn(16, 16)
input2 = torch.randn(16, 16)
input3 = torch.randn(16, 16)
input4 = torch.randn(16, 16)
