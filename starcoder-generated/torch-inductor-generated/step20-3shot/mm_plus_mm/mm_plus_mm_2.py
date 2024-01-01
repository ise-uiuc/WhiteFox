
class Model(nn.Module):
    def forward(self, input):
        t1 = torch.mm(input)
        t2 = torch.mm(input)
        t3 = torch.mm(torch.cat([t1, t2]))
        t4 = torch.mm(torch.cat([t1, torch.mm(t1, t2)]), t3)
        return torch.mm(t3, t4)
# Inputs to the model
input = torch.randn(55, 55)
