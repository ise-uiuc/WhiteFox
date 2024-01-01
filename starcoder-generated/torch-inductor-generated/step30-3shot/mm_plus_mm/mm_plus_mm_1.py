
class Model(torch.nn.Module):
    def forward(self, input1, input2):
        y1 = torch.mm(input1, input1)
        y2 = torch.mm(input2, input2)
        y3 = torch.mm(input1, input2)
        return (y1 + y2) + y3 + y1*y2 + y2*y3 + y2*y2 + y3
# Inputs to the model
input1 = torch.randn(50, 50)
input2 = torch.randn(50, 50)
