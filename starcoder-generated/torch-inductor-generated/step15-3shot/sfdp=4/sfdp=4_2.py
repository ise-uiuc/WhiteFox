
class ScaledDotProductAttention(torch.nn.Module):

    def __init__(self, n_head, n_feat, dropout_rate=0.0):
        super().__init__()
        self.attn_dropout = nn.Dropout(dropout_rate)
 
    def forward(self, query, key, value, attn_mask):
        qk = query @ key.transpose(-2, -1) / math.sqrt(query.size(-1))
        qk = qk + attn_mask
        attn_weight = torch.softmax(qk, dim=-1)
        attn_weight = self.attn_dropout(attn_weight)
        output = attn_weight @ value
        return output

class MultiHeadAttention(torch.nn.Module):

    def __init__(
            self, n_head, n_feat, dropout_rate=0.0, epsilon=1e-6):
        super().__init__()
        self.n_head = n_head
        self.n_feat = n_feat
        self.scaled_dot_attn = ScaledDotProductAttention(
            n_head, n_feat, dropout_rate)
        self.w_q = nn.Linear(n_feat, n_feat, bias=False)
        self.w_k = nn.Linear(n_feat, n_feat, bias=False)
        self.w_v = nn.Linear(n_feat, n_feat, bias=False)
        self.proj = nn.Linear(n_feat, n_feat)
        self.attention_dropout = nn.Dropout(dropout_rate)
        self.layer_norm = nn.LayerNorm(n_feat, eps=epsilon)
 
    def forward(self, query, key, value, attn_mask):
        q, k, v = self.w_q(query), self.w_k(key), self.w_v(value)
        q, k, v = (split(x, self.n_head, self.n_feat) for x in [q, k, v])
        attn_mask = full_attention_mask() if attn_mask is None else attn_mask
        attn_mask = repeat(attn_mask, 'b n -> h b n', h=self.n_head)
        head = self.scaled_dot_attn(q, k, v, attn_mask)
        head = interleave(
            head, self.n_head, self.n_feat)
        head = self.proj(head)
        head = self.attention_dropout(head)
        output = self.layer_norm(query + head)
        return output

class TransformerEncoderLayer(nn.Module):

    def __init__(
            self, n_head, n_feat, ffn_type='linear', ffn_dropout=0.1,
            attention_dropout=0.1, feat_proj_dropout=0.1,
            ffn_activation='relu', layer_norm_eps=1e-6):
        super().__init__()
        self.self_attn = MultiHeadAttention(
            n_head, n_feat, dropout_rate=attention_dropout)
        self.feat_proj = Linear(
            n_feat, n_feat, bias=True, dropout_rate=feat_proj_dropout)
        self.feed_forward = make_ffn(
            n_feat, n_feat, ffn_type, ffn_activation, ffn_dropout)
        self.layer_norm = LayerNorm(n_feat, eps=layer_norm_eps)
 
    def forward(self, x, x_mask):
        _x = self.self_attn(x, x, x, x_mask)
        x = _x + x
        y = self.feat_proj(x) + x
        _y = self.feed_forward(y)
        y = _y + y
        z = self.layer_norm(y)
        return z

class Encoder(nn.Module):

    def __init__(
            self, n_sub_layers, n_head, n_feat, ffn_type='linear',
            num_sources=8, n_block_per_sublayer=1, n_layers=6,
            attention_dropout=0.1, feat_proj_dropout=0.1,
            ffn_dropout=0.1, layer_norm_eps=1e-6):
        super().__init__()
        self.mean_pool = MeanPooling()
        self.conv2d_pool = Conv2dPooling(n_layers, num_sources, n_block_per_sublayer)
        self.input_layer_norm = LayerNorm(n_feat, eps=layer_norm_eps)
        self.subsample_layers = nn.ModuleList([
            nn.ModuleList([EncoderLayer(
                n_head, n_feat, ffn_type, ffn_dropout, attention_dropout,
                feat_proj_dropout, layer_norm_eps)
                for _ in range(n_sub_layers)])
            for _ in range(n_layers)])
        self.final_layer_norm = LayerNorm(n_feat, eps=layer_norm_eps)
 
    def forward(self, padded_input, input_lengths):
        # Input: B x T x D
        input = padded_input.transpose(0, 1)
        # Normalize
        y = self.input_layer_norm(input)
        # Mean pooling
        y = self.mean_pool(y, input_lengths).squeeze(0)
        # Convolution block pooling
        y = self.conv2d_pool(y).squeeze(0)
        # Conv2dPooling output length is always < input length
        input_lengths = None  # To reset the mask computation 
        # 1st subsample layer
        y = self.subsample_layers[0][0](y)
        for sub in self.subsample_layers[0][1:]:
            y_list = repeat(
                y, 'n d -> sub b n d', sub=subsample_rate)
            y_list = chunks(y, y_list, input_lengths)
            y = torch.cat([sub(yi, None) for yi in y_list], dim=0)
            if subsample_rate > 1:
                y_pad_l = y[:, :, 0:subsample_rate-1, :] 
                y_list = split(y, subsample_rate, 1)
                y = torch.cat([
                    F.pad(yi, (0, 0, 0, 1)) if peli else yi for yi, peli in
                    zip(y_list, list(y_pad_l.size(2))[::-1])], 1)
        # Downsampling
        for i in range(1, self.n_layers-1):
            y = self.subsample_layers[i][0](y)
            for sub in self.subsample_layers[i][1:]:
                y_list = repeat(
                y, 'n d -> sub b n d', sub=subsample_rate)
                y_list = chunks(y, y_list, input_lengths)
                y = torch.cat([sub(yi, None) for yi in y_list], dim=0)
                if subsample_rate > 1:
                    y_pad_l = y[:, :, 0:subsample_rate-1, :]
                    y_list = split(y, subsample_rate, 1)
                    y = torch.cat([
                        F.pad(yi, (0, 0, 0, 1)) if peli else yi for yi, peli in
                        zip(y_list, list(y_pad_l.size(2))[::-1])], 1)
        # Last layer normalization
        y = self.final_layer_norm(y)
        return y

class Transformer(nn.Module):

    def __init__(
            self, n_block_per_sublayer, n_layers, num_sources, n_feat, n_head,
            n_sub_layers=2, ffn_activation='relu', ffn_type='linear',
            output_activation=None, attention_dropout=0.1,
            feat_proj_dropout=0.1, ffn_dropout=0.1, embed_dropout=0.0,
            layer_norm_eps=1e-6):
        super().__init__()
        self.encoder = Encoder(
            n_sub_layers, n_head, n_feat, ffn_type, num_sources,
            n_block_per_sublayer, n_layers, attention_dropout,
            feat_proj_dropout, ffn_dropout, layer_norm_eps)
        self.output_layer = Linear(2 * n_feat, n_feat, output_activation)
        self.dropout = nn.Dropout(p=embed_dropout)
        self.embed_layer_norm = nn.LayerNorm(n_feat, eps=layer_norm_eps)
 
    def forward(self, padded_input, input_lengths):
        x = self.dropout(padded_input)
        x = self.encoder(x, input_lengths)
        x = self.embed_layer_norm(x)
        y, z = interleave(x, x, 2), repeat(x, 'd -> b n d', b=padded_input.size(0))
        m = self.output_layer(torch.cat([y, z], dim=2))
        return m

