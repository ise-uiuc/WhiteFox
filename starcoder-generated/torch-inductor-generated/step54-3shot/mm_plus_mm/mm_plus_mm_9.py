
class Model(torch.nn.Module):
    def forward(self, input1, input2, input3, input4, input5):
        y1 = torch.mm(input1, input1)
        y2 = torch.mm(input2, input2)
        y3 = torch.mm(input3, input3)
        y4 = torch.mm(input4, input4)
        y5 = torch.mm(input5, input5)
        y6 = y5 + y4 + y3 + y2 + y1
        return y6
# Inputs to the model
input1 = torch.randn(100, 100)
input2 = torch.randn(100, 100)
input3 = torch.randn(100, 100)
input4 = torch.randn(100, 100)
input5 = torch.randn(100, 100)
