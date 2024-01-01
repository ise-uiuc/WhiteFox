
class Model(torch.nn.Module):
    def forward(self, input1, input2, input3, input4):
        t1 = torch.mm(input1, input2)
        t2 = torch.mm(input3, input4)
        t3 = t1 + t2
        t4 = torch.sigmoid(t3)
        t5 = torch.tanh(t4)
        return t5
# Inputs to the model
input1 = torch.randn(1, 64)
input2 = torch.randn(64, 1)
input3 = torch.randn(64, 64)
input4 = torch.randn(1, 64)
