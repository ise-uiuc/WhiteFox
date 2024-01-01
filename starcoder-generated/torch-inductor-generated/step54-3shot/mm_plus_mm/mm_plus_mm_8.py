
class Model(torch.nn.Module):
    def forward(self, input1, input2):
        y1 = torch.mm(input1, input2)
        return y1
# Inputs to the model
input1 = torch.randn(3, 3)
input2 = torch.zeros(3, 3)
