
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1):
        v1 = torch.mean(x1, dim=1)
        v2 = v1.permute(1, 0).contiguous()
        v3 = v2.view(1)
        return (v1, v2, v3)
# Inputs to the model
x1 = torch.ones(2, device='cuda')
