
class Model(torch.nn.Module):
    def forward(self, query, key, value, input_mask, dropout_p, inv_scale_factor):
        # Expand input_mask with dimensions of the query
        mask = torch.FloatTensor(query.shape[:2]).to(query.device).fill_(1).masked_fill_(input_mask.int(), 0)
        expanded_input_mask = mask.unsqueeze(1).unsqueeze(1)
        # Compute the dot product of the query and the key
        qk = torch.matmul(query, key.transpose(-2, -1))
        # Scale the dot product by the inverse scale factor
        scaled_qk = qk.div(inv_scale_factor)
        # Apply softmax to the scaled dot product
        softmax_qk = scaled_qk.softmax(dim=-1)
        # Apply dropout to the softmax output
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=dropout_p)
        # Compute the dot product of the dropout output and the value
        output = dropout_qk.matmul(value)
        output = output.masked_fill(expanded_input_mask, 0)
        # Return the output
        return output

# Initializing the model
m = Model()

# Inputs to the model
query = torch.randn(4, 8, 4)
key = torch.randn(4, 8, 8)
value = torch.randn(4, 8, 4)
input_mask = torch.tensor([[1,1,0,0], [1,1,0,0], [1,1,1,0], [1,1,0,0]])
dropout_p = 0.
inv_scale_factor = torch.full((8,), 0.5, dtype=key.dtype, device=key.device)
