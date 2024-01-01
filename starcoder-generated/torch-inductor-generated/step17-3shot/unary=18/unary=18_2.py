
class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv1 = torch.nn.Conv1d(1, 1, 7, stride=1, padding=1)
    def forward(self, input_1):
        intermediate = self.conv1(input_1)
        out = torch.sigmoid(intermediate)
        return out
# Input to model
input_1 = torch.randn(1, 1, 64)
