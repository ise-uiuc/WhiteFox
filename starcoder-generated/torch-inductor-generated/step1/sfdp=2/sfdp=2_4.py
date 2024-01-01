
class Model(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embed_tokens = torch.nn.Embedding(config.vocab_size, config.hidden_size)
        self.embed_positions = torch.nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.layers = torch.nn.ModuleList()
        for _ in range(config.num_hidden_layers):
            self.layers.append(torch.nn.TransformerEncoderLayer(config.hidden_size, config.num_attention_heads, dim_feedforward=config.intermediate_size, dropout=config.hidden_dropout_prob, activation=config.hidden_act, normalize_before=False))
        self.dropout = torch.nn.Dropout(config.hidden_dropout_prob)
 
    def forward(self, x):
        v1 = self.embed_tokens(x)
        v2 = self.embed_positions(torch.arange(x.size()[1], dtype=torch.long).unsqueeze(0))
        v3 = v1 + v2
        v4 = v3.transpose(0, 1)
        for layer in self.layers:
            v4 = layer(v4)
        v5 = self.dropout(v4)
        return v5

# Initializing the model with config
config = transformers.RobertaConfig(num_hidden_layers=4, intermediate_size=8, hidden_act="gelu")
m = Model(config)

# Inputs to the model
x = torch.randint(0, 19345, (2, 16))
