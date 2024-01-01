
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.wq = torch.nn.Linear(192, 768)
        self.wk = torch.nn.Linear(192, 768)
        self.wv = torch.nn.Linear(192, 192)
 
    def forward(self, query, key, value, attn_mask):
        q = self.wq(query)
        k = self.wk(key)
        v = self.wv(value)
        v.transpose_(0, 1)
        q = q.unsqueeze(1)
        
        # Compute the dot product of the query and key tensors, and scale it
        _attn_output, attn = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(k.size(-1))
        attn = self.mask_attn_softmax(attn, attn_mask)
        
        # Compute the dot product of the attention weights and the value tensor
        output = torch.matmul(attn, v)
        
        return output, attn

    @staticmethod
    def mask_attn_softmax(attn, attn_mask):
        attn = attn + attn_mask
        attn = torch.nn.functional.softmax(attn, dim=-1)
        return attn

# Initializing the model
model = Model()

batch_size = 1
num_encoder_layers = 1
num_decoder_layers = 1
dim_model = 192
dim_ffn = 3 * dim_model

tgt_len = 5
mem_len = 6

positional_encoding_table = torch.zeros(tgt_len, dim_model)  # tgt_len x dim_model

for pos in range(tgt_len):
    for i in range(dim_model):
        positional_encoding_table[pos][i] = pos * i ** 2

    positional_encoding_table[:, 0::2] = torch.sin(positional_encoding_table[:, 0::2])
    positional_encoding_table[:, 1::2] = torch.cos(positional_encoding_table[:, 1::2])
    positional_encoding_table = positional_encoding_table.unsqueeze(0)

positional_encoding_table = positional_encoding_table.expand(tgt_len, -1, -1).contiguous() # expand(tgt_len, -1, -1)

positional_encoding_table = positional_encoding_table.repeat(batch_size, 1, 1)

# Inputs to the model
query = torch.randn(tgt_len, dim_model)
memory = torch.randn(mem_len, dim_model)
__position_encoding_table__ = positional_encoding_table
__output__, __attention__ = model(query, memory, memory, None)