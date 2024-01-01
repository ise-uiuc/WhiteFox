
class Model(torch.nn.Module):
    def forward(self, input):
        t = torch.mm(input, input)
        t = torch.mm(t, input)
        t = t + torch.mm(t, t)
        t = torch.mm(t, t)
        return t
# Inputs to the model
input = torch.randn(5, 5)
