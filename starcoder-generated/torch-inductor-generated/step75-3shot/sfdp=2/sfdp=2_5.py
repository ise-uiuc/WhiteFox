
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
 
        self.dropout_p = 0.1
        self.softmax_temp = 1./10
        self.dropout = torch.nn.Dropout(self.dropout_p)
        self.softmax = torch.nn.Softmax(dim=-1, dtype=torch.float32)
        self.qat_matmul = torch.npu_multi_head_attention_forward_v2
        
    def forward(self, query, key, value, mask_matrix):
        qk = torch.matmul(query, key.transpose(-2, -1))
        scaled_qk = qk.div(self.softmax_temp)
        attn_weights = self.softmax(scaled_qk)
        attn_masked = attn_weights * mask_matrix
        dropout_attn_masked = self.dropout(attn_masked)
        output = self.qat_matmul(dropout_attn_masked, value)
        return output

# Initializing the model
m = Model()

# Inputs to the model
query = torch.randn(2, 16, 6, 8, 16, 16)
key = torch.randn(2, 16, 25, 26, 16, 16)
value = torch.randn(2, 16, 25, 26, 16, 16)
masking = torch.tril(torch.ones(1, 16, 25, 25).bool().npu())
