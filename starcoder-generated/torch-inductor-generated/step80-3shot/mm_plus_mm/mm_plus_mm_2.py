
class Model(torch.nn.Module):
    def forward(self, input1, input2):
        t1 = torch.mm(input1, input1)
        t2 = torch.mm(input1, input1)
        l1 = [t1, t2]
        sum = 0
        for i in l1:
            sum = i + sum
        t3 = torch.mm(input2, input2)
        return t3 + sum
# Inputs to the model
input1 = torch.randn(8,2)
input2 = torch.randn(8,2)
