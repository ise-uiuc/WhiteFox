
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        x1 = torch.cat((x, x), dim=1)
        x1 = torch.unsqueeze(x1, 0)
        if not x1.size() == (1,10,3,4):
           x1 = x1.resize(1,10,3,4)
        x2 = torch.cat((x1, x1), dim=3)
        x3 = torch.cat((x1, x1), dim=3)
        x4 = torch.cat((x1, x1), dim=3)
        x4_tranposed = x4.transpose(0, 1)
        return torch.cat((x1, x2, x3, x4, x4_tranposed), dim=0).permute(3, 1, 2, 0).contiguous().view(3, -1)
# Inputs to the model
x = torch.randn(2, 3, 4)
