
class Model(torch.nn.Module):
    def forward(self, w1, input1, input2, input3, input4):
        t1 = torch.mm(w1, input3)
        t2 = torch.mm(input2, input3)
        t3 = torch.mm(input1, t1) + torch.mm(t2, input4)
        return t3
# Inputs to the model
input1 = torch.randn(64, 46969)
input2 = torch.randn(46969, 46969)
input3 = torch.randn(46969, 46969)
input4 = torch.randn(46969, 46969)
w1 = torch.randn(1, 46969)
