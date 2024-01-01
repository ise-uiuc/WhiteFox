
class model(torch.nn.Module):
    def __init__(self):
        super(model, self).__init__()
    def forward(self, x1, x2):
        v1 = x1.unsqueeze(0)
        v10 = v1.permute(0, 2, 1)
        v2 = x2.unsqueeze(0)
        v3 = torch.bmm(v10, v2)
        v30 = v3.permute(0, 2, 1)
        v4 = v30.view(4)
        return v4
# Inputs to the model
x1 = torch.randn(4)
x2 = torch.randn(2, 2)
