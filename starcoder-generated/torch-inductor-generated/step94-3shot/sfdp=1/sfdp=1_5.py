
class Model(nn.Module):
    def __init__(self, num_heads, model_dim, num_encoder_layers=16, num_decoder_layers=16, dropout=0.2):
        super(Model, self).__init__()
        self.query = nn.Linear(model_dim, model_dim)
        self.key = nn.Linear(model_dim, model_dim)
        self.value = nn.Linear(model_dim, model_dim)
        self.concat_proj = nn.Linear(model_dim, model_dim)
        self.dropout = nn.Dropout(dropout)
        self.model_dim = model_dim
        self.num_heads = num_heads
        self.num_encoder_layers = num_encoder_layers
        self.num_decoder_layers = num_decoder_layers
        for i in range(num_encoder_layers):
            setattr(self, 'layers_encoder.' + str(i),
                    EncoderLayer(model_dim, num_heads, dropout)
                    )
 
        for i in range(num_decoder_layers):
            setattr(self, 'layers_decoder.' + str(i),
                    DecoderLayer(model_dim, num_heads, dropout)
                    )
        self.post_process_layer = PostProcessLayer(model_dim, dropout)

    def forward(self, inputs):
        