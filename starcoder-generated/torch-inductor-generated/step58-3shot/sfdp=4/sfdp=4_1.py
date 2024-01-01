
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, query_input, key_input, value_input, attn_mask):
        # Apply the following steps (2 in total) on input tensor x:
        # 1. reshape it into [B, 2, 1024] shape
        # 2. add an extra 32 dimension after the batch dimension
        x = query_input @ key_input.transpose(-2, -1) / math.sqrt(query_input.size(-1))
        x = x + attn_mask
        attn_weight = torch.softmax(x, dim=-1)
        output = attn_weight @ value_input
        return output
# Inputs to the model
query_input = torch.randn(1, 4, 2, 1, 1, 1024)
key_input = torch.randn(1, 4, 32, 32, 256)
value_input = torch.randn(1, 4, 32, 32, 256)
attn_mask = (torch.randn(1, 4, 1, 1) > 0).fill_(float('-inf'))
