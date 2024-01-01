
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.dropout = 0.1
        self.heads = 32
        self.seq_len = 256
        self.dim = 64 // self.heads
    def forward(self, input, attn_mask):
        input = torch.dropout(input, 0.2, True)
        q = (input / math.sqrt(input.size(-1))) @ key.transpose(-2, -1)
        qk = q + attn_mask
        # Apply the dropout in the following lines, remember to use dropout to input, query, and key
        attn_weight = torch.dropout(torch.softmax(qk, dim=-1), self.dropout, True)
        output = attn_weight @ value
        # Dropout is not applied here
        return output
# Inputs to the model
input = torch.randn(1, 64, 256, 64)
attn_mask = torch.randn(1, 1, 256, 256)
key = torch.randn(1, 32, 256, 64)
value = torch.randn(1, 32, 256, 64)
