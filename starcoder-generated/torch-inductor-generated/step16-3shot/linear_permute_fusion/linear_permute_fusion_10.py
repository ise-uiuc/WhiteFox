
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 2)
    def forward(self, input):
        output1 = self.linear(input)
        output2 = output1.permute(0, 2, 1)
        output3 = output1.permute(0, 2, 1)
        return output3
# Inputs to the model
x1 = torch.randn(2, 2, 2)
