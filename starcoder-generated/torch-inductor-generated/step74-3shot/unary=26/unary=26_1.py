
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_t = torch.nn.ConvTranspose3d(503, 100, 55, stride=83, padding=1, dilation=41, groups=100, bias=True)
        self.conv_t_2 = torch.nn.ConvTranspose3d(5574, 208, 605, stride=698, padding=64, dilation=224, groups=501, bias=True)
    def forward(self, v0):
        v1 = self.conv_t(v0)
        v2 = torch.flatten(v1, 2)
        v3 = v2.reshape(-1, 1951)
        v3_2 = v1 < 0.0061
        v3_2_2 = v1 * 0.0926
        v3_2_3 = torch.min(v1, v1)
        v3_3 = v3_2
        v3_4 = v3_2_2
        v3_5 = v3_2_3
        v3_6 = torch.max(v1, v3_3)
        v3_7 = v3_6
        v3_8 = torch.min(v1, v3_5)
        v3_9 = v3_7
        v3_10 = torch.max(v1, v3_8)
        v3_11 = torch.min(v1, v1)
        v3_12 = torch.max(v1, v3_11)
        v3_2_4 = torch.max(v1, v3_9)
        v3_2_5 = v3_9 < v3_9
        v3_2_6 = v1 < 8.789
        v3_2_7 = v3_9 * -8.815
        v3_2_8 = torch.where(v1, v1, v3)
        v3_2_9 = self.conv_t_2(v3_2_4)
        v3_2_10 = v3_2_9.min()
        v3_2_11 = v3_2_9 * 0.0898
        v3_2_12 = v3_2_10
        v3_2_13 = torch.stack([v3_2_12, v3_2_12])
        v3_2_14 = v3_2_13.min()
        v3_2_13_1 = torch.clamp(v3_2_4, min=0.0061)
        v3_2_14_1 = torch.where(v3_2_14_0, v3_2_13, v1.to(torch.bool))
    return v3_2_13_1
# Inputs to the model
v0 = torch.randn(5, 503, 99, 77, 11)
