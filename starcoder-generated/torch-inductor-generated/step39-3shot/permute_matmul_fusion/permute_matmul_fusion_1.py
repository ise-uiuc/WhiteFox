
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        return torch.bmm(x.permute(0, 2, 1).detach(), x.permute(0, 2, 1))
# Inputs to the model
x1 = torch.ones(2, 4, 1)[:,:,0]
x2 = torch.ones(1, 2, 4)[:,0,:]
