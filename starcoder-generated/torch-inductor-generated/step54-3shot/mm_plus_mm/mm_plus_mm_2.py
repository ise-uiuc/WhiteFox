
class Model(torch.nn.Module):
    def forward(self, input1, input2):
        y1 = torch.mm(input1, input1)
        y2 = torch.mm(input2, input2)
        y3 = torch.mm(input1, input2)
        y4 = torch.mm(input2, input1)
        y5 = y2 + y3 + y4 + y1
        return y5
# Inputs to the model
input1 = torch.randn(10, 40)
input2 = torch.randn(10, 40)
