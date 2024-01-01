
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose = torch.nn.ConvTranspose2d(3, 8, 3, stride=2, padding=1, dilation=2)
    def forward(self, x1):
        v1 = self.conv_transpose(x1)
        v1_add_3_res = v1 + 3
        v2_cmax_res = torch.clamp(v1_add_3_res, min=0)
        v3_cmin_res = torch.clamp(v2_cmax_res, max=6)
        v4_mul_res = v1 * v3_cmin_res
        v5_div_res = v4_mul_res / 6
        return v5_div_res
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
