
class Model(torch.nn.Module):
    def forward(self, query, key, value, key_padding_mask, attn_mask, memory_mask, dropout_p):
        qk = torch.matmul(query, key.transpose(-2, -1))
        scaled_qk = qk.div(0.32)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=dropout_p)
        output = dropout_qk.matmul(value)
        return output

# Inputs to the model
query = torch.randn(2, 5, 3, 512)
key = torch.randn(2, 5, 128, 36)
value = torch.randn(2, 5, 128, 36)
key_padding_mask = torch.randn(2, 5, 128, 36).to(dtype=torch.bool)
attn_mask = torch.randn(2, 5, 512, 512).to(dtype=torch.bool)
memory_mask = torch.randn(2, 5, 23, 36).to(dtype=torch.bool)
dropout_p = 0.2
