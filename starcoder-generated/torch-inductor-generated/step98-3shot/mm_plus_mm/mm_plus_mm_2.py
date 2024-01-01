
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.mat_mul = torch.bmm
    def forward(self, input1, input2):
        t1 = self.mat_mul(input1.view(2, 1, -1), input2.view(2, -1, 1), torch.Size([2, 2]))
        t2 = self.mat_mul(input1.view(2, 1, -1), input2.view(2, 1, -1), torch.Size([2, 2]))
        return t1 + t2
# Inputs to the model
input1 = torch.randn(1, 6, 20)
input2 = torch.randn(1, 20, 3)
