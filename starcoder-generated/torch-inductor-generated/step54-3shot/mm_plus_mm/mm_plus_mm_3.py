
class Model(torch.nn.Module):
    def forward(self, input1):
        y1 = torch.mm(input1, input1)
        y2 = y1[:13]
        y3 = y2 + y2
        y4 = y2 * y3
        return y4
# Inputs to the model
input1 = torch.randn(25, 25)
