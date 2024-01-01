
class MultiHeadAttention(torch.nn.Module):
    def __init__(self, head_num, input_size, dropout_p):
        super(MultiHeadAttention, self).__init__()
 
        hidden_size = input_size // head_num
        self.head_num = head_num
        self.head_size = input_size // head_num
 
        self.query_projection = torch.nn.Linear(input_size, head_num*hidden_size, bias=False)
        self.key_projection = torch.nn.Linear(input_size, head_num*hidden_size, bias=False)
        self.value_projection = torch.nn.Linear(input_size, head_num*hidden_size, bias=False)
        self.output_projection = torch.nn.Linear(head_num*hidden_size, input_size)
 
        self.dropout = torch.nn.Dropout(dropout_p)
 
    def forward(self, query, key, value, mask=None):
        batch_size, q_len, _ = query.size()
        _, k_len, _ = key.size()
        _, v_len, _ = value.size()
 
        if mask is not None:
            mask = mask.unsqueeze(1)
 
        # head_num * batch_size * q_len * head_size
        query = self.query_projection(query).view(self.head_num, batch_size, q_len, self.head_size)
        key = self.key_projection(key).view(self.head_num, batch_size, k_len, self.head_size)
        value = self.value_projection(value).view(self.head_num, batch_size, v_len, self.head_size)
 
        # head_num * batch_size * q_len * k_len
        score = torch.matmul(query, key.transpose(-2, -1))
        score = score / np.sqrt(self.head_size)
 
        if mask is not None:
            mask = mask.to(score.dtype)
            score = score.masked_fill(mask == 0, -np.inf)
 
        # attn_qk: head_num * batch_size * q_len * k_len
        attn_qk = torch.nn.functional.softmax(score, dim=-1)
        attn_qk = self.dropout(attn_qk)
 
        # head_num * batch_size * q_len * k_len * head_size 
        attn_qk = attn_qk.transpose(-2, -1).repeat(1, 1, 1, 1, v_len)
        # head_num * batch_size * q_len * k_len * head_size * v_len
        value = value.repeat(self.head_num, 1, 1, 1, 1)
        
        # out: head_num * batch_size * q_len * head_size * v_len
        out = torch.matmul(attn_qk, value)
        # out: batch_size * q_len * head_num * head_size * v_len
        out = out.permute(1, 2, 0, 3, 4).contiguous().view(batch_size, q_len, -1)
        
        return self.output_projection(out)

# Initializing the model
head_num = 8
input_size = 128
dropout_p = 0.1
m = MultiHeadAttention(head_num, input_size, dropout_p)

# Inputs to the model
query = torch.randn(128, 100, input_size)
key = torch.randn(128, 200, input_size)
value = torch.randn(128, 200, input_size)
