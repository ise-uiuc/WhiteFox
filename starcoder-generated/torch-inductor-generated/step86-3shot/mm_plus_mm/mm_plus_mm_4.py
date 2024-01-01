
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.mat_mul = torch.mm
    def forward(self, input1, input2, input3):
        t1 = self.mat_mul(input1, input2)
        t2 = self.mat_mul(input3, input1)
        return t1 + t2
# Inputs to the model
input1 = torch.randn(32, 32)
input2 = torch.randn(32, 32)
input3 = torch.randn(32, 32)
