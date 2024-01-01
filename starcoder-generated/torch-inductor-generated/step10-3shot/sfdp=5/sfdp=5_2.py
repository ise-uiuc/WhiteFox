
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder_layer1 = torch.nn.TransformerEncoderLayer(288, 8, 2048, 128, 0.0)
        self.encoder_layer2 = torch.nn.TransformerEncoderLayer(288, 8, 2048, 128, 0.0)
        self.encoder = torch.nn.TransformerEncoder(self.encoder_layer1, 1)
        self.encoder2 = torch.nn.TransformerEncoder(self.encoder_layer2, 1)
        self.decoder = torch.nn.TransformerDecoder()
        self.decoder_layer1 = torch.nn.TransformerDecoderLayer(288, 8, 2048, 128, 0.0)
        self.decoder = torch.nn.TransformerDecoder(self.decoder, self.decoder_layer1)
 
    def forward(self, x1):
        v1 = x1 @ self.encoder_layer1.self_attn.out_proj.weight.t()
        v1 = v1.reshape(1, 1, 288, 768)
        v1 = v1 + self.encoder_layer1.self_attn.bias[:, :, None, None]
        v1 = v1.permute(0, 2, 1, 3)
        v1 = v1.reshape(1, -1, 288)
        v2 = torch.gelu(v1)
        v2 = v2.permute(0, 2, 1)
        v2 = v2.reshape(1, -1, 288, 768)
        v = self.encoder(v2)
        v3 = x1 @ self.encoder_layer2.self_attn.out_proj.weight.t()
        v3 = v3.reshape(1, 1, 288, 768)
        v3 = v3 + self.encoder_layer2.self_attn.bias[:, :, None, None]
        v3 = v3.permute(0, 2, 1, 3)
        v3 = v3.reshape(1, -1, 288)
        v4 = torch.gelu(v3)
        v4 = v4.permute(0, 2, 1)
        v4 = v4.reshape(1, -1, 288, 768)
        v5 = self.encoder2(v4)    
        v6 = v5 @ self.decoder_layer1.self_attn.out_proj.weight.t()
        v6 = v6.reshape(1, 1, 288, 768)
        v6 = v6 + self.decoder_layer1.self_attn.bias[:, :, None, None]
        v6 = v6.permute(0, 2, 1, 3)
        v6 = v6.reshape(1, -1, 288)
        v7 = torch.gelu(v6)
        v7 = v7.permute(0, 2, 1)
        v7 = v7.reshape(1, -1, 288, 768)
        v8 = self.decoder(v7, v5)
        return v8

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 288, 141)
