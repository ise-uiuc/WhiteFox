
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = torch.nn.Transformer([
                torch.nn.TransformerEncoderLayer(d_model=256, nhead=8),
            ],
            num_encoder_layers=6
        )
 
    def forward(self, x1):
        v1 = self.model(x1, src_key_padding_mask)[0]
        return v1

# Initializing the model
device = torch.device("cuda")
model = Model().to(device)

# Input to the model
mask_token = torch.cuda.LongTensor([255])
x1 = torch.rand(1, 32, 256).fill_(mask_token).long().to(device)
input_size = x1.size()[:]
