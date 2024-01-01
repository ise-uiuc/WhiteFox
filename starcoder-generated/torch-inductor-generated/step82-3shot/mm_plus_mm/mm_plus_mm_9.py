
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, input1, input2, input3, input4):
        t1 = torch.addmm(input1, input2, input3, beta=0.0, alpha=1.0)
        t2 = torch.addmm(input1, input4, input3, beta=0.0, alpha=1.0)
        t3 = t1 + t2
        return t3
# Inputs to the model
input1 = torch.randn(2, 3)
input2 = torch.randn(2, 3)
input3 = torch.randn(2, 3)
input4 = torch.randn(2, 3)
