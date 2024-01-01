
class Model(torch.nn.Module):
    def forward(self, input0, input1, input2, input3):
        t0 = torch.mm(input0, input1)
        t1 = torch.mm(input2, input3)
        t2 = torch.mm(t0, t1)
        return torch.mm(t0, t1)
# Inputs to the model
input0 = torch.randn(1, 1)
input1 = torch.randn(5, 5)
input2 = torch.randn(5, 5)
input3 = torch.randn(5, 5)
