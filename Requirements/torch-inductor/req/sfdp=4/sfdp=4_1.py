qk = query @ key.transpose(-2, -1) / math.sqrt(query.size(-1)) # Compute the dot product of the query and key, and scale it
qk = qk + attn_mask # Add the attention mask to the scaled dot product
attn_weight = torch.softmax(qk, dim=-1) # Apply softmax to the result
output = attn_weight @ value # Compute the dot product of the attention weights and the value
