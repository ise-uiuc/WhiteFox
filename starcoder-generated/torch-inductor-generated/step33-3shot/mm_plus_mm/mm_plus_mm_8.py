
class Model(torch.nn.Module):
    def forward(self, input1, input2):
        t1 = torch.mm(input1, input2)
        t2 = torch.sigmoid(t1)
        t3 = torch.sigmoid(t2)
        t4 = t3 * t1
        return t4
# Inputs to the model
input1 = torch.randn(10, 10)
input2 = torch.randn(10, 10)
