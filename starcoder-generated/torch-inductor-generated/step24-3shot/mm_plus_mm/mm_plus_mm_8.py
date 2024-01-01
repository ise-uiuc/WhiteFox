
class Model(torch.nn.Module):
    def forward(self, input):
        t1 = torch.mm(input, torch.rand(40, 40))
        t2 = torch.mm(input, torch.rand(32, 32))
        t3 = torch.mm(input, torch.rand(32, 32))
        return torch.mm(torch.mm(t1, t3), torch.rand(32, 32))
# Inputs to the model
input1 = torch.nn.Module
