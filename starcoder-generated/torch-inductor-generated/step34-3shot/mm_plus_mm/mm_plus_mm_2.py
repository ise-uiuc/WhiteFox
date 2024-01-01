
class Model(torch.nn.Module):
    def forward(self, input1, input2, input3, input4):
        y1 = torch.mm(input1, input1)
        y2 = torch.mm(input2, input2)
        y3 = torch.mm(input3, input3)
        y4 = torch.mm(input4, input4)
        y5 = y1 + y2 + y3 + y4
        return y5
# Inputs to the model
input1 = torch.randn(50, 50)
input2 = torch.randn(50, 50)
input3 = torch.randn(50, 50)
input4 = torch.randn(50, 50)
