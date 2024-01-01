
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.op1 = torch.nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=(1, 1), stride=(2,2), padding=0)
    def forward(self, X):
        v1 = self.op1(X)
        return v1
# Inputs to the model
X = torch.randn(1, 1024, 4, 4)
