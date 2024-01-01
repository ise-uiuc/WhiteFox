
class Model(torch.nn.Module):
    def __init__(
        self,
        embedding_dim,
        num_heads,
        hidden_dim,
        dropout_p,
        num_hidden_layers
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.dropout_p = dropout_p
        self.num_hidden_layers = num_hidden_layers
 
        self.attn_layers = []
        for layer_idx in range(num_self_attention_layers):
            layer = MultiHeadAttentionLayer(
                embedding_dim,
                num_heads,
                hidden_dim,
                dropout_p
            )
            self.attn_layers.append(layer)
 
        self.ff_layers = torch.nn.ModuleList([
            TransformerFeedFowardLayer(
                embedding_dim,
                hidden_dim,
                dropout_p
            ) for _ in range(num_hidden_layers)
        ])
 
    def attention(self, x):
        for attn_layer in self.self_attention_layers:
            x = attn_layer(x)
        return x
 
    def forward(self, x):
        for ff_layer in self.ff_layers:
            x = ff_layer(x)
        return x

# Initializing the model
m = Model(512, 8, 2048, 0.1, 5)

# Inputs to the model
x = torch.randn(1, 512, 8, 8)
