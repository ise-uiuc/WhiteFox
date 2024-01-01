
class SinkCat(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear2 = torch.nn.Linear(20, 20)
        self.linear1 = torch.nn.Linear(20, 30)
    def forward(self, x):
        batchsize = x.shape[0]
        t = torch.ones(batchsize, 20)
        x = torch.cat((x, t), dim=0)
        x = self.linear1(x).relu()
        x = x.view(batchsize, 5, -1)
        x = x.permute(0, 2, 1)
        x = torch.cat((x, x), dim=1).view(batchsize, 5, -1)
        x = x.permute(0, 2, 1)
        return x
# Inputs to the model
x = torch.randn(4, 20, requires_grad=True)
