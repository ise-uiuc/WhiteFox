
class Model(torch.nn.Module):
    def forward(self, input0):
        m1 = input0 + torch.mm(input0, input0)
        return torch.mm(m1, m1) + input0

# Inputs to the model
input1 = torch.randn(5, 5)
