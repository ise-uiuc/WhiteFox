
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 11, 1, stride=1, padding=0)
    def forward(self, x1):
        v1 = self.conv(x1)
        v1_2 = v1.clone().detach().requires_grad_()
        v2 = 3 + v1
        v3 = torch.clamp_min(v2, 0)
        v3_2 = v3.clone().detach().requires_grad_()
        v4 = torch.clamp_max(v3, 6)
        v4_2 = v4.clone().detach().requires_grad_()
        v5 = torch.nn.functional.conv2d(v3_2, v4_2, v1_2, padding=1, padding_mode='zeros')
        v6 = v1_2 + 0.5
        return v6
# Inputs to the model
x1 = torch.randn(2, 3, 64, 64)
