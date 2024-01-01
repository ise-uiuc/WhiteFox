
class Model(torch.nn.Module):
    def __init__(self, d_model=64, n_head=8, dropout_p=0.1, n_layer=2):
        super().__init__()
        self.n_layer = n_layer
        self.embed = torch.nn.Embedding(28, d_model)
        self.pos_encode = PositionalEncoding(d_model, dropout_p)
        self.encoder_layer = EncoderLayer(d_model, n_head, dropout_p)
        self.encoder = Encoder(self.encoder_layer, n_layer)
 
    def forward(self, x, query):
    
        seq_len = x.size(1)
        embed = self.embed(x) * math.sqrt(self.d_model)
        embed = self.pos_encode(embed)
            
        attn_mask = subsequent_mask(seq_len, attn_mask=False).cuda().bool() # Get an ascending mask
        output = self.encoder(embed, attn_mask, src_key_padding_mask=None)[0] # Apply the transformer encoder module to the embedded tensor and get the ouput of the module
        return torch.bmm(output, query.unsqueeze(2)).squeeze(2) # Compute the dot product of the query and output of the transformer encoder module, and squeeze the 2-D shape of the dot-producted result

# Initializing the model
m = Model()

# Inputs to the model
x = torch.randint(1, 28, (1, 10)).cuda()
query = torch.rand(1, 64, 1).cuda()
