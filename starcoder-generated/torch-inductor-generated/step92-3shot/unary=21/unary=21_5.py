
class ModelTanh(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv3d(3, 64, 3, stride=(2, 2), padding=(1,1,0), dilation=(1,1,1))
    def forward(self, x):
        v1 = self.conv1(x)
        v2 = torch.tanh(v1)
        return v2
# Inputs to the model
x = np.random.randn(1, 3, 64, 64, 64)
x = torch.from_numpy(x)
