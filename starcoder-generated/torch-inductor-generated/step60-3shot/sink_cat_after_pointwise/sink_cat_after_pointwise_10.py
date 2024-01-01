
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        t1 = torch.cat([torch.ones(1, 20), torch.ones(1, 20)], dim=0).view(10, 20)
        t2 = torch.ones(10, 20)
        # This will cause "y" to be a tensor with rank 0
        y = torch.cat((t1, t2), dim=1)
        return y.view(y.shape[0], -1).relu()
# Inputs to the model
x = torch.ones(1, 20)
