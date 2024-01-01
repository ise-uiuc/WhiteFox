
class TransformerModel(torch.nn.Module):
    def __init__(self, cfg, d_model, nhead, num_layers=1, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.attention = nn.MultiheadAttention(d_model, nhead)
        self.norm = nn.LayerNorm(d_model)
        self.norm1 = nn.LayerNorm(d_model)
        # self.encoder = nn.Sequential(
        #     nn.Conv2d((cfg.INPUT_QUANTIZATION_BITWIDTH - 1), 64, 1, stride=1, padding=1),
        #     nn.BatchNorm2d(64),
        #     nn.ReLU(),
        #     nn.Conv2d(64, 128, 1, stride=1, padding=1),
        #     nn.BatchNorm2d(128),
        #     nn.ReLU(),
        #     nn.Conv2d(128, 512, 1, stride=1, padding=1),
        #     nn.BatchNorm2d(512),
        #     nn.ReLU(),
        #     nn.Conv2d(512, 2048, 1, stride=1, padding=1),
        #     nn.BatchNorm2d(2048),
        #     nn.ReLU(),
        # )
        self.encoder = nn.Conv2d(64, 2048, 1)
        self.head = nn.Conv2d(d_model, 2048, 1)


    def _forward(self, src, mask=None, quantize=False):

        src1 = self.norm(src)
        # src1 = self.conv(x)
        # src =
        q = k = self.encoder(src1)
        # src_mask = mask
        # if not self.training:
        #     for i in range(cfg.BERT_LAYER):
        #         # for j in range(len(q)):
        #         q[i] = q[i]/cfg.MULTIPLIER
        #         k[j] = k[j]/cfg.MULTIPLIER
        # q = (q/128).round()*128
        # k = (k/128).round()*128
        attn_output, attn_output_weights = self.attention(q, k, q, attn_mask=mask, need_weights=False)
        attn_output = self.dropout(attn_output)
        src = src + attn_output
        src = self.norm1(src)
        src = self.head(src)
        src = src.permute([0,2,3,1])
        src = self.dropout1(src)
        # src = self.fc(src)
        # src = torch.flatten(src, start_dim=1)
        # src = src / cfg.MULTIPLIER
        # src = (src/128).round()*128
        return src, attn_output_weights
 
    def forward(self, src):
        x, mask = src[0], src[1]
        x, o = self._forward(x, mask, quantize=False)
 
        return [x]

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
 
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        #pe = pe.expand(1, max_len, d_model)
        self.register_buffer('pe', pe)
 
    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class TokenClassification(nn.Module):

    # Initialize your model here
    def __init__(self, cfg, d_model, tagset_size, ngram_size, pretrain_num_epochs, encoder_type, pretrain_device, pretrain_weights=None):
        super(TokenClassification, self).__init__()

        # Specify encoder: BERT/ALBERT/ROBERTA
        self.encoder_type = encoder_type
        if encoder_type == 'BERT':
          self.encoder = BertModel.from_pretrained(cfg.PRETRAIN_TYPE)
        else: # ALBERT/ROBERTA
          self.encoder = AutoModel.from_pretrained(cfg.PRETRAIN_TYPE)

        # 69068
        self.pool = nn.AdaptiveAvgPool2d(output_size=1)
        # self.pool1 = nn.AdaptiveAvgPool2d(output_size=1)
        # self.pool2 = nn.AdaptiveMaxPool2d(output_size=1)
        # self.pool3 = nn.AdaptiveMaxPool2d(output_size=1)
        self.head = BertModel(cfg.BERT_HIDDEN_SIZE,  cfg.BERT_HIDDEN_SIZE)


        dff = cfg.BERT_HIDDEN_SIZE

        # self.conv = torch.nn.Conv2d(4, 1280, 1, stride=1, padding=1)
       
        self.fc = torch.nn.Linear(32, tagset_size)

    # Passages will be passed in as a torch tensor of shape (B, T)
    def forward(self, x):
        #x = x.detach()
        #x = torch.zeros(1, 100, 512)
        #print(x)
        if self.encoder_type == 'BERT':

          # inputs_embeds = self.encoder.embeddings(x)

          input_ids = torch.tensor([[0, 1, 2], [5, 12, 4]]) # Batch size 2
          token_type_ids = torch.tensor([[0, 0, 1], [2, 0, 1]]) # Batch size 2

          input_ids = torch.tensor([[0, 1, 2]]) # Batch size 1
          token_type_ids = torch.tensor([[0, 0, 1]]) # Batch size 2

          attention_mask = torch.tensor([[1, 1, 1], [1, 1, 1]]) # Batch size 2
          head_mask = [None]*4

          embedding = self.encoder.embeddings(input_ids, token_type_ids=token_type_ids)


          encoded_layers, pooled_output= self.encoder(embedding.attention_mask)




          # a = torch.tensor([[1.0,1.0,1.0]])
          # a = a**16
          # a = a.round()
          # a = a**-16
          # print(a)
          #print(encoded_layers[0])
          #x = self.head(encoded_layers[1])

        else:
          
          outputs = self.encoder(x)
          outputs1 = outputs['pooler_output']
        #   print(outputs1.shape)
        # for p in parameters:
        #   print(f"{p.numel()}\t{p.size()}")

        # out = outputs['pooler_output'].squeeze().reshape(1, -1)
        x = F.max_pool2d(outputs1.unsqueeze(1) + outputs1.unsqueeze(3), (3,3)) + outputs1.unsqueeze(1) + outputs1.unsqueeze(3) - outputs1.unsqueeze(4)* outputs1.unsqueeze(2) - outputs1.reshape(x.shape[0], x.shape[1]//3, 3, x.shape[2], 3)
        # x = x.reshape(x.shape[0], x.shape[1], 3)
        # x = x.abs()

          x = outputs1.permute(0, 2, 3, 1)
          # x = self.conv(x)
          x = torch.flatten(x, start_dim=1)
          # x = self.pool(x)
          out = self.fc(x)
          return (x.unsqueeze(1), out)


# Passages will be passed in as a torch tensor of shape (B, T)
#     def forward(self, x):
#         # x = x.detach()
#         # x = torch.zeros(1, 100, 512)
#         # print(x)
#         outputs, outputs1 = self.encoder(x)

#         # x = self.conv(x)
#         x = outputs
#         # outputs1 = outputs['pooler_output']
#         # x = self.pool(outputs)
#         # x = self.fc1(outputs1)

#         # for p in parameters:
#         #     print(f"{p.numel()}\t{p.size()}")

#         # x = self.conv(x)
#         # x = torch.flatten(x, start_dim=1)
#         # x = self.fc(x)
#         x = x.permute(0,2,3,1)
#         # x = outputs1.permute(0, 2, 3, 1)
#         x = torch.flatten(x, start_dim=1)
#         #outputs1 = outputs1.squeeze().reshape(1, -1)
#         x = self.fc(x)
#         # x = self.pool(x)

#         return (x.unsqueeze(1), outputs1)
