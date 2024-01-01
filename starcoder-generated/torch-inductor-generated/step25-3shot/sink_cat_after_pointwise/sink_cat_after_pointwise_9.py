
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        aaa = x.view(x.shape[0], -1)
        if aaa.dim() < 2:
            aaa = torch.relu(x)
        else:
            aaa = aaa.tanh()
        bbb = aaa.view(aaa.shape[0], -1).tanh()
        if aaa.size(1)!= x.shape[1] * 2:
            bbb = torch.relu(bbb)
        return bbb
# Inputs to the model
x = torch.randn(5, 3, 4)
