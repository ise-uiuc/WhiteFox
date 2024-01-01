
class Model(torch.nn.Module):
    def forward(self, input, input1):
        t1 = torch.mm(input1, input)
        t1 = torch.mm(input, input)
        v1 = t1.mm(input)
        t2 = torch.mm(t1, input)
        t2 = torch.mm(input, input)
        v2 = t2.add(input)
        t3 = t2.mm(t2)
        return (5*t3) / v1
# Inputs to the model
input = torch.randn(10, 10)
input1 = torch.randn(10, 5)
