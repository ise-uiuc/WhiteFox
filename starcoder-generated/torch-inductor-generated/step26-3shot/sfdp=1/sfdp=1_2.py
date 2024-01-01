
class Model(torch.nn.Module):
    def __init__(self, dropout_p, emb_s, num_encoder_layers, embed_dim):
        super().__init__()
        self.dropout_p = dropout_p
        self.emb_s = emb_s
        self.num_encoder_layers = num_encoder_layers
        self.embed_dim = embed_dim
        initrange = 0.1
        self.scale_factor = 1 / (self.embed_dim**0.5)
        embed_matrix = torch.empty(num_embeddings=vocab_size, embedding_dim=embed_dim)
        torch.nn.init.uniform_(embed_matrix, -initrange, initrange)
        self.embed_matrix = torch.nn.parameter.Parameter(embed_matrix)
 
    def forward(self, input_ids):
        embed = self.embed_matrix(input_ids)
        embed = embed * self.emb_s
        embed = embed.div(self.embed_dim**0.5)
        self.embed = embed
        embed = self.embed_dropout(embed)
        embed = self.encoder(embed)
 
    def embed_dropout(self, embed):
        embed = torch.nn.functional.dropout(embed, p=self.dropout_p)
        return embed
 
    def encoder(self, embed):
        for i in range(self.num_encoder_layers):
            attn_norm = torch.clone(embed)
            query = self.multihead_attn_query(embed)
            key = self.multihead_attn_key(embed)
            value = self.multihead_attn_value(embed)
            out = self.attn_dropout(query * key * value.transpose(-2, -1))
            out = out.reshape(out.size(0), out.size(1), out.size(2)*out.size(3))
            embed = embed + self.encoder_layer_norm_2(out)
            mlp = self.mlp(embed)
            mlp = mlp.reshape(mlp.size(0), mlp.size(1), mlp.size(2)*mlp.size(3))
            embed = embed + self.encoder_layer_norm_2(mlp)
        return embed
 
class Decoder(nn.Module):
    def __init__(self, dropout_p, embeddings_scale, num_decoder_layers, embed_dim):
        super().__init__()
        self.dropout_p = dropout_p
        self.emb_s = embeddings_scale
        self.num_decoder_layers = num_decoder_layers
        self.embed_dim = embed_dim
        initrange = 0.1
        self.scale_factor = 1 / (self.embed_dim**0.5)
        embed_matrix = torch.empty(num_embeddings=vocab_size, embedding_dim=embed_dim)
        torch.nn.init.uniform_(embed_matrix, -initrange, initrange)
        self.embed_matrix = torch.nn.parameter.Parameter(embed_matrix)
 
    def forward(self, encoder_output, input_ids):
        embed = self.embed_matrix(input_ids)
        embed = embed * self.emb_s
        embed = embed.div(self.embed_dim**0.5)
        embed = self.embed_dropout(embed)
        embed = self.decoder(embed)
 
    def embed_dropout(self, embed):
        embed = torch.nn.functional.dropout(embed, p=self.dropout_p)
        return embed
 
    def decoder(self, embed):
        for i in range(self.num_decoder_layers):
            attn_norm = torch.clone(embed)
            query = self.multihead_attn_query(embed)
            key = self.multihead_attn_key(attn_norm)
            value = self.multihead_attn_value(attn_norm)
            out = self.attn_dropout(query * key * value.transpose(-2, -1))
            attn_norm = attn_norm + self.decoder_layer_norm_1(out)
            mlp = self.mlp(attn_norm)
            embed = embed + self.decoder_layer_norm_2(mlp)
        return embed
 
# Initializing the model
model = Model(dropout_p=dropout_p, emb_s=emb_s, num_encoder_layers=num_encoder_layers, embed_dim=embed_dim)
decoder_model = Decoder(dropout_p=decoder_dropout_p, embeddings_scale=emb_s, num_decoder_layers=num_decoder_layers, embed_dim=embed_dim)
 
# Inputs to the model
input_ids = torch.randint(0, vocab_size, (x2.size(0), x.size(1)))
 
