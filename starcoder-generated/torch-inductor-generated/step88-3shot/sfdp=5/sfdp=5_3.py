
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.heads = 96
        self.seq_len = 160
        self.dim = 512 // self.heads
        self.embed_dim = 512
        self.fc =torch.nn.Linear(784,512)
        self.fc1 = torch.nn.Linear(512,1568)
        self.fc2 = torch.nn.Linear(1568,2048)
        self.fc3 = torch.nn.Linear(2048,2048)
        self.fc4 = torch.nn.Linear(2048,512)
    def forward(self, query, key, value, attn_mask):
        qk = query @ key.transpose(-2, -1) / math.sqrt(query.size(-1))
        qk = qk + attn_mask
        attn_weight = torch.softmax(qk, dim=-1)
        attn_weight = torch.dropout(attn_weight, 0.00, True)
        output = attn_weight @ value
        return output
# Inputs to the model
query = torch.randn(1, 1568, 80)
key = torch.randn(1, 1568, 80)
value = torch.randn(1, 1568, 80)
attn_mask = torch.randn(1, 1, 80, 80)
