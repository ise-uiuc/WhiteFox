
class Model(torch.nn.Module):
    def forward(self, input1, input2):
        e1 = torch.exp(input1)
        e2 = torch.exp(input2)
        p1 = e1 / (e1 + e2)
        p2 = e2 / (e1 + e2)
        return p1 + p2
# Inputs to the model
input1 = torch.randn(16, 16)
input2 = torch.randn(16, 16)
