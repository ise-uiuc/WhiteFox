
class Model(torch.nn.Module):
    def forward(self, input):
        t1 = torch.mm(input, input)
        t2 = torch.mm(t1, input)
        t3 = torch.mm(input, t2)
        return (t1 + t2 + t3) / 6
# Inputs to the model
input1 = torch.randn(2, 3)
input2 = torch.randn(3, 2)
input3 = torch.randn(2, 2)
input4 = torch.randn(3, 3)
