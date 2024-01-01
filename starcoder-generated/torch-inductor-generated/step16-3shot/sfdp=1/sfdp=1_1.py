
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
 
    def forward(self, q, k, v, query_mask, key_padding_mask):
        scale_factor = 1 + key_padding_mask.max(dim=-1, keepdim=True).values.abs()
        scaled_qk = torch.matmul(q, k.transpose(-2, -1))/scale_factor.unsqueeze(-1)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=dropout_p)
        output = torch.matmul(dropout_qk, v)
        return output

# Initializing the model
m = Model()

# Inputs to the model
q, k, v = torch.randn(1,8,1,8), torch.randn(1,8,2,8), torch.randn(1,8,2,8)
query_mask, key_padding_mask = torch.zeros(1,8,1,1), torch.ones(1,8,2,1)
