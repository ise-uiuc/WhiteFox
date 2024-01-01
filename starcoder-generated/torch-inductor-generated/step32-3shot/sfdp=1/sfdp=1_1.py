
class Model(torch.nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.w_1 = torch.nn.Linear(d_model, d_model)
        self.softmax = torch.nn.Softmax(dim=-1)
        self.dropout = torch.nn.Dropout(0.1)
        self.w_2 = torch.nn.Linear(d_model, d_model)
    
    def forward(self, enc_outputs, memory_key_padding_mask=None):
        attn = self.w_1(enc_outputs)
        attn = self.softmax(attn)
        attn = self.dropout(attn)
        if memory_key_padding_mask is not None:
            attn = attn.masked_fill(memory_key_padding_mask, float("-inf"))
        attnT = torch.bmm(attn, enc_outputs)
        attnT = self.w_2(attnT)
        return attnT

# Initializing the model
m = Model(256)

# Inputs to the model
enc_outputs = torch.randn(2048, 256)
memory_key_padding_mask = torch.randint(2, size=(2048, 256), dtype=torch.int64)
