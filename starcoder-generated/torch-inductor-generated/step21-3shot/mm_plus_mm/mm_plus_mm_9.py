
class Model(torch.nn.Module):
    def forward(self, input1, input2, input3, input4):
        t1 = torch.mm(input3, input4)
        t2 = torch.mm(input1, input4)
        t3 = t1 + t2
        return t3
# Inputs to the model
t1 = torch.randn(10, 5, 5)
t2 = torch.randn(5, 10, 10)
t3 = t1 + t2
