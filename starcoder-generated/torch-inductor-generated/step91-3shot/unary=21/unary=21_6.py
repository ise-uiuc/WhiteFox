
class ModelTanh(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, input1):
        v1 = torch.tanh(input1)
        v2 = torch.neg(v1)
        return v2
# Inputs to the model
input1 = torch.randn(55296)
