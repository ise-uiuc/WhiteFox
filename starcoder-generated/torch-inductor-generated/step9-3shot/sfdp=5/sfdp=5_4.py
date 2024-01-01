
class Model(torch.nn.Module):
    def __init__(self, d_model, n_head, d_k, d_v, dropout_p, input_tensor):
        super().__init__()
        
        self.encoder = torch.nn.Sequential()
        self.encoder.add_module('l1',
                                 torch.nn.Linear(in_features=np.prod(input_tensor.shape[1:]), out_features=d_model))
        self.encoder.add_module("l1_dropout",torch.nn.Dropout(dropout_p))
        
        # Add positional embedding to the encoder output.
        self.pos_encoder = PositionalEncoding()
    
        # Add encoder layers.
        self.transformer = EncoderLayer(d_model, n_head, d_k, d_v, dropout_p)
        
        self.decoder = torch.nn.Sequential()
        self.decoder.add_module('l2',
                                 torch.nn.Linear(in_features=d_model, out_features=np.prod(input_tensor.shape[1:])))
 
        # Add positional embedding to the decoder output.
        self.pos_decoder = PositionalEncoding()
        self.decoder.add_module("l2_dropout",torch.nn.Dropout(dropout_p))

        # Add decoder layers.
        self.transformer2 = DecoderLayer(d_model, n_head, d_k, d_v, dropout_p)
    
    def forward(self, query, key, value, attn_mask):
        #print("input",query.shape,key.shape,value.shape,attn_mask.shape)
        encoder_out = self.encoder(query)  
        #print("encoder",encoder_out.shape)
        encoder_out = self.pos_encoder(encoder_out)
        #print("pos_encoder",encoder_out.shape)
        encoder_out = self.transformer(encoder_out, key, value, attn_mask)
        #print("encoder2",encoder_out.shape)
        decoder_out = self.decoder(encoder_out)
        #print("decoder1",decoder_out.shape)
        decoder_out = self.pos_decoder(decoder_out)
        #print("decoder2",decoder_out.shape)
        decoder_out = self.transformer2(decoder_out, key, value, attn_mask)
        #print("decoder3",decoder_out.shape)
        return decoder_out

# Initializing the model
m = Model(d_model=512, n_head=8, d_k=64, d_v=64, dropout_p=0.1, input_tensor=x)
print(m)

# Inputs to the model
query = torch.randn(1, 64, 512)
key = torch.randn(64, 8, 64)
value = torch.randn(64, 8, 64)
attn_mask = torch.ones(512, 512)

