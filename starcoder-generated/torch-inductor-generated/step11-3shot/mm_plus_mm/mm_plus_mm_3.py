
class Model(torch.nn.Module):
    def forward(self, input):
        t1 = torch.mul(input, input)
        t2 = torch.mul(input, input)
        t3 = t1.add(t2)
        return t3
# Inputs to the model
input = torch.randn(5, 5, dtype=torch.float64)
