
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, input1, input2):
        t1 = input1 + 5
        t2 = input2 + 5
        t3 = input1 * t2
        return t3
# Inputs to model
input1 = torch.randn(20, 20)
input2 = torch.randn(20, 20)
