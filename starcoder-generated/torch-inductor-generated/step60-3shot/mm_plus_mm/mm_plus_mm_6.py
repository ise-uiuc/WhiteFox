
class Model(torch.nn.Module):
    def forward(self, input1, input2, input3, input4):
        t0, t1 = torch.max(input3, dim=1)
        t1 = torch.cat([t1, t0], dim=1)
        t0 = torch.mm(input1, input2)
        t3 = torch.mm(input4, input2)
        t2 = torch.mm(t1, input1)
        return input1 * t2 + input4 * t3
# Inputs to the model
input1 = torch.randn(5, 5)
input2 = torch.randn(5, 5)
input3 = torch.randn(5, 5)
input4 = torch.randn(5, 5)
