
class Model(torch.nn.Module):
    def forward(self, x1):
        v1 = x1.permute(0, 2, 1)
        batch_size, num_channels, num_features = v1.size()
        v2 = torch.nn.functional.max_pool2d(v1, kernel_size=[3, num_features], stride=[3, num_features]) # Permute the 3rd (dim=2) and 4th (dim=3) dimensions of the `x`
        v3 = torch.mul(v1, v2) # Element-wise multiplication
        return v3 if batch_size > -1 else torch.nn.functional.adaptive_avg_pool1d(v3, output_size=6)
# Inputs to the model
x1 = torch.randn(1, 4, 4)
