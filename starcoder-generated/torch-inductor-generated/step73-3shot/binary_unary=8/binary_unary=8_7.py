
class Model(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv1 = torch.nn.Conv1d(6, 2, kernel_size=(2,), stride=(2,), padding=(1,), dilation=(1, ))
    def forward(self, x):
        v1 = self.conv1(x)
        v2 = self.conv1(x)
        v3 = v1 + v2
        v4 = torch.relu(v3)
        return v4
# Inputs to the model
x = torch.randn(1, 6, 49, 1)
