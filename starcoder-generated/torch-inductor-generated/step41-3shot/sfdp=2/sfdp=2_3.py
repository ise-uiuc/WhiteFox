
class Model(torch.nn.Module):
    def __init__(self, num_heads, num_classes, d_input_dim, d_kv):
        super().__init__()
        self.num_heads = num_heads
        self.num_classes = num_classes
        self.d_kv = d_kv
        self.d_head = d_kv // num_heads

        self.attn_q = torch.nn.Linear(d_input_dim, d_input_dim)
        self.attn_k = torch.nn.Linear(d_input_dim, d_input_dim)
        self.attn_v = torch.nn.Linear(d_input_dim, d_input_dim)

        self.class_proj = torch.nn.Linear(d_input_dim, num_classes)

    def forward(self, query, key, value, dropout_rates):
        Q = self.attn_q(query)
        K = self.attn_k(key)
        V = self.attn_v(value)
        Q = Q.reshape(Q.shape[0], -1, self.num_heads, self.d_head).permute([0, 2, 1, 3])
        K = K.reshape(K.shape[0], -1, self.num_heads, self.d_head).permute([0, 2, 1, 3])
        V = V.reshape(V.shape[0], -1, self.num_heads, self.d_head).permute([0, 2, 1, 3])

        attn_logits = torch.matmul(Q, K.transpose(-2, -1)) / self.d_head**0.5
        inv_scale_factor = (1. / Q.shape[-1])**0.5
        dropout_probs = 1 - dropout_rates
        attn_weights = F.dropout(F.softmax(attn_logits * inv_scale_factor, dim=-1), p=dropout_probs, training=True)
        attn_output = torch.matmul(attn_weights, V)
        attn_output = attn_output.transpose(1, 2).reshape(shape=[-1, Q.shape[1], self.num_heads * self.d_head])
        logits = self.class_proj(attn_output)
        return logits

# Initializing the model
m = Model(num_heads=4, num_classes=1024, d_input_dim=1024, d_kv=1024)

# Inputs to the model
x_query = torch.randn(size=(1, 512, 1024))
x_key = x_query
x_value = x_query
dropout_rates = 0.1
