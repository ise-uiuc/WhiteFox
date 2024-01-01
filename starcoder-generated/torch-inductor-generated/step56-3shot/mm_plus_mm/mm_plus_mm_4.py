
class Model(torch.nn.Module):
    def forward(self, input1, input2):
        t1 = torch.mm(input1, input2)
        t2 = torch.mm(input1, input1)
        if torch.sum(input1) > 100:
            t3 = torch.mm(input1, input1)
        else:
            t3 = torch.mm(input2, input1)
        t4 = t2 + t3
        return t4
# Inputs to the model
input1 = torch.randn(3, 3)
input2 = torch.randn(3, 3)
