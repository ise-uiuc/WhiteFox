
class Model(torch.nn.Module):
    def forward(self, input, weights):
        t1 = torch.mm(input[0], weights[0])
        t2 = torch.mm(input[1], weights[1])
        t3 = t1 + t2
        return t3
# Inputs to the model
input = [torch.randn(3, 3), torch.randn(3, 3)]
weights = [torch.randn(3, 3), torch.randn(3, 3)]
