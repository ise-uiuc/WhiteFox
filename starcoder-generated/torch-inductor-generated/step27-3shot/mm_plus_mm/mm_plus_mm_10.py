
class TorchModel(nn.Module):
    def __init__(self):
        super(TorchModel, self).__init__()
    def forward(self, input):
        v = torch.unsqueeze(input, 2)
        return v.sum(1)
# Inputs to the model
input1 = torch.randn(3, 4)
input2 = torch.randn(3, 4)
input3 = torch.randn(3, 4)
input4 = torch.randn(3, 4)
