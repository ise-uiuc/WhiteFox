
class Model(torch.nn.Module):
    def __init__(self, query_num, key_num, value_num):
        super().__init__()
 
    def forward(self, q, k, v, attn_mask):
        qk = np.matmul(q, k.transpose(1,0,2))
        qk = qk / math.sqrt(k.size(-1))
        qk = qk + attn_mask
        attn_weight = torch.softmax(qk, dim=-1)
        output = torch.matmul(attn_weight, v)
        return output