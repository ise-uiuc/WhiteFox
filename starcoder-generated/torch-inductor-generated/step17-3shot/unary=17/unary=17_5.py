
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose = torch.nn.ConvTranspose2d(3, 32, 3, padding=1, stride=1)
    def forward(self, x1):
        v1 = torch.nn.functional.interpolate(x1, scale_factor=1.0, recompute_scale_factor=False)
        v2 = v1.contiguous()
        v3 = v2.float()
        v4 = v3.to(torch.float16)
        v5 = v4.to(torch.float32)
        v6 = v5.to(torch.float64)
        v7 = v6.to(torch.int8)
        v8 = v7.to(torch.int16)
        v9 = v8.to(torch.int32)
        v10 = v9.to(torch.int64)
        v11 = v10.to(torch.bool)
        v12 = v11.to(torch.complex64)
        v13 = v12.to(torch.complex128)
        v14 = v13.to(torch.qint8)
        v15 = v14.to(torch.quint8)
        v16 = v15.to(torch.qint32)
        v17 = v13.to(torch.bfloat16)
        return v17
# Inputs to the model
x1 = torch.randn(1, 3, 224, 224)
