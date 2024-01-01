
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 2).cuda()
    def forward(self, x1):
        v3 = x1
        v1 = torch.nn.functional.relu(self.linear(x1))
        v2 = v1.permute(0, 2, 1)
        return (v1, v2)
# Inputs to the model
x1 = torch.randn(5, 2, 2, device='cuda')
