
class Model(torch.nn.Module):
    def forward(self, in1, in2, in3, in4):
        temp = (in1 * in2) + (in3 * in4)
        return torch.mm(temp, temp)
# Inputs to the model
in1 = torch.randn(2, 2)
in2 = torch.randn(2, 2)
in3 = torch.randn(2, 2)
in4 = torch.randn(2, 2)
