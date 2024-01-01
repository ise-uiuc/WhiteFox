
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.emb_dim = 256
        self.num_heads = 2
        self.attention_head_size = int(self.emb_dim / self.num_heads)
        self.all_head_size = self.num_heads * self.attention_head_size
        self.query_proj = torch.nn.Linear(self.emb_dim, self.all_head_size)
        self.key_proj = torch.nn.Linear(self.emb_dim, self.all_head_size)
        self.value_proj = torch.nn.Linear(self.emb_dim, self.all_head_size)
        self.out_proj = torch.nn.Linear(self.all_head_size, self.emb_dim)
 
    def forward(self, q, k, v, attention_mask=None):
        q = self.query_proj(q)
        k = self.key_proj(k)
        v = self.value_proj(v)
 
        q = q / math.sqrt(q.size(-1))
        q = q.view(q.size(0), q.size(1), self.num_heads, self.attention_head_size).permute(0, 2, 1, 3)
        k = k.view(k.size(0), k.size(1), self.num_heads, self.attention_head_size).permute(0, 2, 1, 3)
        v = v.view(v.size(0), v.size(1), self.num_heads, self.attention_head_size).permute(0, 2, 1, 3)
        attn_mask = (attention_mask == 0).to(torch.float) # For masking out the irrelevant tokens
        attn_mask = attn_mask.view(attn_mask.size(0), -1)
        attn_mask = attn_mask.unsqueeze(1).unsqueeze(2)
        attn_mask = attn_mask.repeat(1, self.num_heads, q.size(1), 1)
        attn_mask = attn_mask.view(-1, q.size(1), k.size(-1))
        output = torch.matmul(q, k)
        output = output + attn_mask
        output = F.softmax(output, dim=-1)
        output = output.view(q.size(0), self.num_heads, q.size(1), k.size(-1))
        output = output.permute(0, 2, 1, 3)
        output = output.contiguous().view(output.size(0), output.size(1), -1)
        output = self.out_proj(output)
        return output
 
 
def generate_single_training_sample(batch_size):
    src_input = torch.empty(batch_size, 16, 256).uniform_(0, self.vocab_size)
    tgt_input = torch.empty(batch_size, 16, 256).uniform_(0, self.vocab_size)
    src_mask, tgt_mask = generate_mask(src_input, tgt_input)
    tgt_input = [token.item() for token in tgt_input[0].view(-1)]
    return src_input.long().to(self.device), tgt_input, src_mask, tgt_mask
 
# Initializing the model
m = Model()
# Inputs to the model
x1 = torch.randn(1, 16, 256)
x2 = torch.randn(1, 16, 256)
