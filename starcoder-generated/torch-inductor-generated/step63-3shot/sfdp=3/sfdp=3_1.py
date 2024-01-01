
class Model(torch.nn.Module):
    def forward(self, __input_tensor__):
        query = torch.rand(1, 512, 64, 14)
        key = torch.rand(1, 512, 64, 14)
        value = torch.rand(1, 512, 64, 256)
        scale_factor = torch.rand(query.size(1), query.size(1), device=query.device)
        dropout_p = 0.1
        v1 = torch.matmul(query, key.transpose(-2, -1))
        v2 = v1.mul(scale_factor)
        v3 = v2.softmax(dim=-1)
        v4 = torch.nn.functional.dropout(v3, p=dropout_p)
        v5 = torch.matmul(v4, value)
        return v5

 