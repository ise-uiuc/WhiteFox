
class Model(torch.nn.Module):
    def forward(self, input, inputy):
        m1 = torch.mm(input, input)
        m2 = torch.mm(input, inputy)
        m3 = torch.mm(inputy, input)
        m4 = torch.mm(inputy, inputy)
        m5 = m1 * m2
        m6 = m3 * m4
        return m1+m2+m3+m4+m5+m6
# Inputs to the model
input = torch.randn(5, 5)
inputy = torch.randn(5, 5)
