
class Model(torch.nn.Module):
    def forward(self, input1, input2, input3):
        y1 = torch.mm(input1, input2)
        y2 = torch.mm(input1, input2)
        y3 = torch.mm(input1, input3)
        y4 = torch.mm(input2, input3)
        return y1, y2, y3, y4
# Inputs to the model
input1 = torch.randn(3, 3)
input2 = torch.randn(3, 3)
input3 = torch.randn(3, 3)
