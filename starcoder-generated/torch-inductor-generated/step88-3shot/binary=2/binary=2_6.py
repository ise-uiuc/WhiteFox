
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv1d(10, 2, kernel_size=(4,), stride=(1,), padding=(1,))
    def forward(self, input):
        v1 = self.conv(input)
        v2 = v1 - 1.467261972230284
        return v2
# Inputs to the model
input = torch.randn(20, 10, 52)
