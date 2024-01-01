
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # Please initialize the parameters to the same value.
        self.linear1 = torch.nn.Linear(5,5, bias=False)
        self.linear1.weight = torch.nn.Parameter(torch.ones_like(self.linear1.weight))
        self.linear2 = torch.nn.Linear(5,5, bias=False)
        self.linear2.weight = torch.nn.Parameter(torch.ones_like(self.linear2.weight))
    
    def forward(self, x):
        x1 = torch.nn.functional.relu(x)
        v1 = torch.repeat_interleave(x1, 5, dim=0)
        v1 = torch.cat((v1,x1), dim=1)
        v1 = v1.permute(2, 0, 1)
        x2 = torch.nn.functional.relu(self.linear1(v1[0,...]))
        x3 = torch.nn.functional.relu(self.linear1(v1[1,...]))
        v2 = torch.cat([x2, x3], dim=0)
        x4 = torch.nn.functional.relu(self.linear2.forward(v2))
        x5 = torch.max(x4, dim=1)[0]
        return x5
# Inputs to the model
x1 = torch.randn(2, 2)
