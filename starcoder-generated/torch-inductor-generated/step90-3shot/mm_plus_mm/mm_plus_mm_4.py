
class Model(nn.Module):
    def forward(self, input1):
        t1 = torch.mm(input1, input1)
        return torch.mm(t1, t1.mm(t1))
# Inputs to the model
input1 = torch.randn(3)
