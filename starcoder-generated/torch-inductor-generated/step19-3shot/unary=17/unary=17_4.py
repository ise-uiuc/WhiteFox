
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 32, kernel_size=7, padding=(0, 0), stride=(1, 1), dilation=(1, 1))
    def forward(self, x1, x2):
        # Apply batch normalization
        v1 = 1.0
        v2 = bn_ops.fused_batch_norm(x1, running_mean=v1, running_var=v1, weight=v1, bias=v1, training=False, momentum=v1, eps=v1)
        # Batch normalization ends
        v3 = torch.nn.functional.interpolate(x2, scale_factor=0.25, mode='bilinear', align_corners=False)
        v4 = self.conv(v2)
        v5 = v3.add(v4)
        v6 = torch.relu(v5)
        v7 = torch.max(v6, torch.tensor(v6, dtype=torch.float))
        v8 = torch.dropout(v7, p=0.5, training=False)
        return v8
# Inputs to the model
x1 = torch.randn(1, 3, 28, 28, dtype=torch.float32)
x2 = torch.randn(1, 32, 4, 4, dtype=torch.float32)
