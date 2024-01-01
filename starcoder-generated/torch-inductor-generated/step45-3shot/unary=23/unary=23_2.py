
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose = torch.nn.ConvTranspose2d(1, 3, 1, stride=1, padding=0)
    def forward(self, x1, x2):
        v1 = x1.transpose(dim0=0, dim1=2)
        v2 = self.conv_transpose(v1)
        v3 = v2.transpose(dim0=0, dim1=2)
        # v4 = torch.bmm(x2, v3)
        v4 = torch.einsum('n i c h w, m c h w -> n m i', (v3, x2))
        # v5 = torch.bmm(x2, torch.bmm(v2, v3))
        v5 = torch.bmm(x2, torch.einsum('n c h w, c h w -> n c h w', (v3, v3)))
        return v4, v5
# Inputs to the model
x1 = torch.randn(8, 1, 8, 8)
x2 = torch.randn(8, 1, 8, 8)
