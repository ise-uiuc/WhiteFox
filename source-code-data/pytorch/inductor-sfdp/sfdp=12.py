attn_weight = torch.bmm(query, key.transpose(1, 2)).softmax(dim=-1)
attn_weight = torch.nn.functional.dropout(attn_weight, p=dropout_p)
torch.bmm(attn_weight, value)