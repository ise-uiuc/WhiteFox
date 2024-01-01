
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.query = torch.nn.Parameter(torch.randn(1, 32, 1, 1))
        self.key = torch.nn.Parameter(torch.randn(1, 32, 1, 1))
        self.value = torch.nn.Parameter(torch.randn(1, 32, 1, 1))
        self.dropout_p = 0.4

    def forward(self, qk_input, attn_mask_input):
        qk = torch.matmul(qk_input + self.query, torch.transpose(self.key, -2, -1)) / math.sqrt(32)
        qk = qk + attn_mask_input
        attn_weight = torch.softmax(qk, dim=-1)
        attn_weight = torch.dropout(attn_weight, p, True)
        output = torch.matmul(attn_weight, self.value)
        return output

# Initializing the model
m = Model()

# Inputs to the model
qk_input = torch.randn(1, 32, 1, 1)
attn_mask_input = torch.ones(1, 1, 1, 1)
