qk = query @ key.transpose(-2, -1) / math.sqrt(query.size(-1)) # Compute the dot product of the query and key, and scale it
qk = qk + attn_mask # Add the attention mask to the scaled dot product
attn_weight = torch.softmax(qk, dim=-1) # Apply softmax to the result
attn_weight = torch.dropout(attn_weight, dropout_p, True) # Apply dropout to the softmax output
output = attn_weight @ value # Compute the dot product of the dropout output and the value
