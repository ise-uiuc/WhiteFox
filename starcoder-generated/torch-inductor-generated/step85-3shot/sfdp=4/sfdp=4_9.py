
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, input, mask):
        Q4 = nn.softmax(input, dim=-1)
        output = Q4 * 1.1
        return output
# Inputs to the model
input = torch.randn(3,3)
mask = (torch.rand(3,3) > 0.7).fill_(-1000000000.0)
