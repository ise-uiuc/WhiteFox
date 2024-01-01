
class Model(torch.nn.Module):
    def forward(self, query, key, value):
        qk = query @ key.transpose(-2, -1)
        inv_scale_factor = self.scale_factor.rsqrt()
        scaled_qk = qk * inv_scale_factor
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=self.dropout_p)
        ouput = dropout_qk @ value
        return output

# Initializing the model
m = Model(dropout_p=0.5)

# Inputs to the model
query = torch.randn(2, 5, 16, 24)
key = torch.randn(2, 3, 8, 24)
value = torch.randn(2, 3, 8, 24)
