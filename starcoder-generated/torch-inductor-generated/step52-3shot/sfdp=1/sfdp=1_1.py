
class ScaledDotProductAttention(nn.Module):
    def __init__(self, dropout, **unused):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
 
    def forward(self, q_tensor, k_tensor, v_tensor):
        # Computing the dot product of the query and key tensors
        qk_tensor = torch.matmul(q_tensor, k_tensor.transpose(-2, -1))
 
        # Scale the dot product
        scale_factor = k_tensor.size(-1) ** 0.5
        inv_scale_factor = 1. / scale_factor
        scaled_dot_product_tensor = qk_tensor * inv_scale_factor
 
        # Apply softmax to the scaled dot-product tensor along the last axis
        softmax_qk_tensor = F.softmax(scaled_dot_product_tensor, dim=-1)
 
        # Apply dropout to the softmax output
        dropout_qk_tensor = self.dropout(softmax_qk_tensor)
 
        # Compute the dot product of the dropout output and the value tensor
        output_tensor = torch.matmul(dropout_qk_tensor, v_tensor)
 
        return output_tensor

 
class BertSelfAttention(nn.Module):
    def __init__(self, num_hidden_layers, **unused):
        super().__init__()
        self.attention = nn.ModuleList([
            ScaledDotProductAttention(dropout) for _ in range(num_hidden_layers)
        ])
 
    def forward(self, hidden_states):
        for attention_layer in self.attention:
            hidden_states = attention_layer(hidden_states, hidden_states, hidden_states)
        return hidden_states


# Initializing the model
m = BertSelfAttention(1)

# Inputs to the model
hidden_states = torch.randn(6, 128, 32, 4096)
