
class Model(torch.nn.Module):
    def forward(self, input):
        t1 = torch.matmul(input, input)
        t2 = torch.matmul(input, input)
        t3 = t1 + t2
        return t3
# Inputs to the model
input = torch.randn(3, 3)
