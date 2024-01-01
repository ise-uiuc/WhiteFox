
class PatternDetection_Relu_Concat_View(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        t1 = torch.cat((x,x), dim=1)
        t2 = torch.concat((x,x), dim=1).view(-1, 3) # (t2 is equivalent of t)
        t3 = torch.sum(t2, dim=1) if (t3.shape[0] is 1) else torch.relu(t2)
        return t3
# Inputs to the model
x = torch.randn(3, 2, requires_grad=True)
