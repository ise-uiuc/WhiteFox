
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.input0 = torch.randn((1, 5))
    def forward(self, input1):
        output4 = torch.mm(input1, self.input0)
        return output4
# Inputs to the model
input1 = torch.randn((1, 5))
