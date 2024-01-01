
class Model(torch.nn.Module):
    def forward(self, input1, input2):
        t1 = torch.mm(input1, input1)
        t2 = torch.mm(input2, input2)
        return torch.mm(t1, t2)
# Inputs to the model
input1 = torch.randn(16, 32)
input2 = torch.randn(16, 32)
