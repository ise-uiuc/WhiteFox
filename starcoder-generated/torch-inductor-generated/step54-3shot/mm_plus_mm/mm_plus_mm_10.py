
class Model(torch.nn.Module):
    def forward(self, input1):
        y1 = torch.mm(input1, input1)
        return y1
# Inputs to the model
input1 = torch.randn(100, 100)
