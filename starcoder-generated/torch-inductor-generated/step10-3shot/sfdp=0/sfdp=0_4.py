
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.key1 = torch.nn.Parameter(torch.randn(7))
        self.key2 = torch.nn.Parameter(torch.randn(3, 4, 5))
        self.key3 = torch.nn.Parameter(torch.randn(1, 2, 3, 4, 5))
    def forward(self, x1):
        q = x1.transpose(-2, -1)
        k1 = self.key1
        k2 = self.key2
        k3 = self.key3
        if math.isnan(q.numel()):
            inv_scale1 = math.sqrt(k1.size(0))
            inv_scale2 = math.sqrt(k2.size(0))
            inv_scale3 = math.sqrt(k3.size(1))
        else:
            inv_scale1 = math.sqrt(k1.size(0))
            inv_scale2 = math.sqrt(k2.size(1))
            inv_scale3 = 1.0
        scaled_dot_product = (q * k1).sum(-1) / inv_scale1
        scaled_dot_product = scaled_dot_product.unsqueeze(1)
        scaled_dot_product = scaled_dot_product + (q * (k2.transpose(-2, -1))).sum(-1) / inv_scale2
        if math.isnan(scaled_dot_product[0][0].numel()):
            scaled_dot_product = scaled_dot_product[:, :, None]
        else:
            scaled_dot_product = scaled_dot_product.unsqueeze(2)
        scaled_dot_product = scaled_dot_product.unsqueeze(3)
        if math.isnan(scaled_dot_product[0][0][0][0].numel()):
            scaled_dot_product = scaled_dot_product[:, :, None, :, :, None]
        else:
            scaled_dot_product = scaled_dot_product.unsqueeze(4)
        scaled_dot_product = scaled_dot_product + (q * (k3.transpose(-1, -2))).sum(-1) / inv_scale3
        scaled_dot_product = scaled_dot_product.permute(0, 2, 3, 1, 4, 5).contiguous()
        output = scaled_dot_product.view(1, 1, 64, 64)
        return output

# Inputs to the model
x1 = torch.randn(7, 64, 64)
