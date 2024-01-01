
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.heads = 5
        # Set attention window size to be 13 x sequence length
        self.window = 13
        self.seq_len = 128
        self.dim = 776 // self.heads
    def forward(self, query, key, value, attn_mask):
        # Split the attention window horizontally.
        h_split_query = torch.split(query, self.window, 2)
        h_split_key = torch.split(key, self.window, 2)
        h_split_value = torch.split(value, self.window, 2)

        # Split the attention window vertically.
        # We need to make sure padding is NOT applied to the first (top) split
        v_split_query = torch.split(h_split_query[-1], self.window, 3)
        v_split_key = torch.split(h_split_key[-1], self.window, 3)
        v_split_value = torch.split(h_split_value[-1], self.window, 3)

        # Concat the first (top) split vertically with the other left splits.
        h_split_query = torch.cat([h_split_query[-1], *v_split_query[:-1]], dim=2).flatten(0, 1)
        h_split_key = torch.cat([h_split_key[-1], *v_split_key[:-1]], dim=2).flatten(0, 1)
        h_split_value = torch.cat([h_split_value[-1], *v_split_value[:-1]], dim=2).flatten(0, 1)

        # Flatten the attention window to be only over the tokens (i.e. removing heads, dim, width)
        h_split_query = h_split_query.view(-1, h_split_query.size(2), h_split_query.size(-1)//16)
        h_split_key = h_split_key.view(-1, h_split_key.size(2), h_split_key.size(-1)//16)
        h_split_value = h_split_value.view(-1, h_split_value.size(2), h_split_value.size(-1)//16)

        # Query-key attention (aka scaled dot product attention)
        qk = h_split_query @ h_split_key.transpose(-2, -1) / math.sqrt(h_split_query.size(-1))
        qk = qk + attn_mask

        # Softmax over the height of the window
        qk = qk.view(-1, self.window//3, self.window)
        qk = qk.transpose(2, 1)
        attn_weight = torch.softmax(qk, dim=-1)

        # We need to make sure padding is NOT applied to the first (top) split
        attn_weight = attn_weight.view(-1, self.window//3, self.window//5)
        attn_weight = attn_weight.transpose(2, 1)

        attn_weight = torch.dropout(attn_weight, 0.8, True)
        # Weighted sum (using height as a broadcast dim)
        attn_weight = attn_weight.view(-1, 1, self.window)
        output = (attn_weight * h_split_value).sum(1)
        return output
# Inputs to the model
query = torch.randn(1, 1, 128, 776)
key = torch.randn(1, 1, 128, 776)
value = torch.randn(1, 1, 128, 776)
attn_mask = torch.randn(1, 1, 128, 128)
