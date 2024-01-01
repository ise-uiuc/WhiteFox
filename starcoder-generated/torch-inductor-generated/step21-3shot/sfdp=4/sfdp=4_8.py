
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input1, input2, input3):
        v1 = input1 @ input2.transpose(-2, -1)
        v2 = v1 / math.sqrt(v1.size(-1))
        v3 = v2 + input3
        v4 = torch.softmax(v3, dim=-1)
        output = v4 @ input3
        return output

# Initializing the model
m = Model()

# Inputs to the model
input1 = torch.randn(3, 4, 5)
input2 = torch.randn(5, 6)
input3 = torch.randn(6, 4)
