
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.mat_mul = torch.mm
    def forward(self, input1, input2):
        t1 = self.mat_mul(input1, input1)
        t2 = self.mat_mul(input2, input2)
        t3 = self.mat_mul(input1, input2)
        return t1 + t2 + t3
# Inputs to the model
input1 = torch.randn(5, 5)
input2 = torch.randn(5, 5)
