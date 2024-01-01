
class Model(torch.nn.Module):
    def forward(self, input1):
        t1 = torch.mm(input1, torch.mm(input1, input1))
        t2 = torch.mm(input1, torch.mm(input1, torch.mm(input1, input1)))
        return t1 + t2 + t1
# Inputs to the model
input1 = torch.randn(8, 8)
