
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(in_features=64, out_features=1024, bias=True)
 
    def forward(self, x4):
        v7 = self.linear(x4)
        v8 = torch.sigmoid(v7)
        v9 = v7 * v8
        