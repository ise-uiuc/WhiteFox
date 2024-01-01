
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x, w0, w1, w2):
        y = torch.cat((x, w0.view(w0.size(0), -1)), dim=1)
        y = y + w1.view(w1.size(0), -1)
        return y.tanh() + w2.view(w2.size(0), -1).tanh()
# Inputs to the model
w0 = torch.randn(64, 4096, 7, 7)
w1 = torch.randn(64, 4096, 1, 1)
w2 = torch.randn(64)
x = torch.randn(64, 3, 224, 224)
