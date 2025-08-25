# transformer.py
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                             (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        ## PERBAIKAN: Ubah shape pe agar bisa di-broadcast ke [Time, Batch, Dim]
        pe = pe.unsqueeze(1) # Shape menjadi [max_len, 1, d_model]
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        x: [Time, Batch, d_model]
        """
        ## PERBAIKAN: Sesuaikan dengan format [Time, Batch, Dim]
        # self.pe[:x.size(0)] akan mengambil slice [Time, 1, d_model]
        # yang bisa ditambahkan ke x via broadcasting
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)

def generate_square_subsequent_mask(sz: int, device):
    mask = torch.triu(torch.full((sz, sz), True, dtype=torch.bool, device=device), diagonal=1)
    return mask

class TransformerNMT(nn.Module):
    def __init__(self,
                 src_vocab_size: int,
                 trg_vocab_size: int,
                 d_model: int = 256,
                 nhead: int = 8,
                 num_encoder_layers: int = 4,
                 num_decoder_layers: int = 4,
                 dim_feedforward: int = 1024,
                 dropout: float = 0.1,
                 pad_id: int = 0,
                 bos_id: int = 1,
                 eos_id: int = 2,
                 device: torch.device = torch.device("cpu")):
        super().__init__()
        self.device = device
        self.pad_id = pad_id
        self.bos_id = bos_id
        self.eos_id = eos_id
        self.d_model = d_model

        self.src_embed = nn.Embedding(src_vocab_size, d_model, padding_idx=pad_id)
        self.trg_embed = nn.Embedding(trg_vocab_size, d_model, padding_idx=pad_id)
        self.pos_enc = PositionalEncoding(d_model, dropout)

        # PASTIKAN batch_first=False
        self.transformer = nn.Transformer(
            d_model=d_model, nhead=nhead,
            num_encoder_layers=num_encoder_layers, num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward, dropout=dropout,
            batch_first=False, norm_first=False
        )
        self.generator = nn.Linear(d_model, trg_vocab_size)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None: nn.init.zeros_(m.bias)
            if isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, mean=0, std=d_model**-0.5)

    @torch.no_grad()
    def make_pad_mask(self, seq):
        # seq inputnya [Time, Batch] -> return [Batch, Time]
        return (seq == self.pad_id).t().contiguous()

    def forward(self, src, trg, teacher_forcing_ratio: float = 0.0):
        """
        src: [Tsrc, B], trg: [Ttrg, B]
        """
        src_key_padding_mask = self.make_pad_mask(src)
        tgt_key_padding_mask = self.make_pad_mask(trg)

        src_emb = self.pos_enc(self.src_embed(src) * math.sqrt(self.d_model))
        tgt_emb = self.pos_enc(self.trg_embed(trg) * math.sqrt(self.d_model))

        ## PERBAIKAN: Gunakan size(0) untuk panjang sekuens, bukan size(1)
        tgt_mask = generate_square_subsequent_mask(trg.size(0), src.device)

        mem = self.transformer.encoder(src_emb, src_key_padding_mask=src_key_padding_mask)
        out = self.transformer.decoder(
            tgt=tgt_emb, memory=mem,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=src_key_padding_mask
        ) # out: [Ttrg, B, C]

        logits = self.generator(out)  # [Ttrg, B, V]
        return logits, None

    @torch.no_grad()
    def beam_search_decode(self, src, beam_size=5, max_len=50, length_penalty=0.7):
        """
        src: [Tsrc, B]
        returns: ys: [Tout, B]
        """
        B = src.size(1)
        if B != 1:
            raise NotImplementedError("Beam search saat ini hanya diimplementasikan untuk batch size = 1.")

        src_key_padding_mask = self.make_pad_mask(src)
        src_emb = self.pos_enc(self.src_embed(src) * math.sqrt(self.d_model))
        memory = self.transformer.encoder(src_emb, src_key_padding_mask=src_key_padding_mask)

        # Inisialisasi beam
        # `beams` adalah list berisi tuple: (sequence, log_prob_score)
        initial_beam = (torch.full((1, 1), self.bos_id, dtype=torch.long, device=src.device), 0.0)
        beams = [initial_beam]
        
        completed_hypotheses = []

        for _ in range(max_len):
            new_beams = []
            for seq, score in beams:
                # Jika sekuens sudah selesai dengan <eos>, jangan diekspansi lagi
                if seq[-1, 0].item() == self.eos_id:
                    completed_hypotheses.append((seq, score))
                    continue

                # Dapatkan output dari token terakhir
                tgt_emb = self.pos_enc(self.trg_embed(seq) * math.sqrt(self.d_model))
                tgt_mask = generate_square_subsequent_mask(seq.size(0), src.device)
                
                out = self.transformer.decoder(
                    tgt=tgt_emb, 
                    memory=memory,
                    tgt_mask=tgt_mask
                ) # out: [T_current, 1, C]
                
                logits = self.generator(out[-1, :, :]) # logits: [1, V]
                log_probs = F.log_softmax(logits, dim=-1).squeeze(0) # [V]

                # Dapatkan top-k token berikutnya dan skornya
                top_log_probs, top_indices = torch.topk(log_probs, beam_size)

                for i in range(beam_size):
                    next_tok_id = top_indices[i].item()
                    log_prob = top_log_probs[i].item()
                    
                    new_seq = torch.cat([seq, torch.tensor([[next_tok_id]], device=src.device)], dim=0)
                    new_score = score + log_prob
                    new_beams.append((new_seq, new_score))

            # Sortir semua kandidat beam baru berdasarkan skor
            # dan ambil k terbaik
            beams = sorted(new_beams, key=lambda x: x[1], reverse=True)[:beam_size]
            
            # Cek jika semua beam teratas sudah selesai
            if all(b[0][-1,0].item() == self.eos_id for b in beams):
                completed_hypotheses.extend(beams)
                break

        # Jika ada hipotesis yang belum selesai (mencapai max_len)
        completed_hypotheses.extend(beams)

        # Normalisasi skor dengan length penalty untuk menghindari favoritisme kalimat pendek
        # score / (sequence_length ^ length_penalty)
        best_hypothesis = max(
            completed_hypotheses, 
            key=lambda x: x[1] / (x[0].size(0) ** length_penalty)
        )
        
        best_seq = best_hypothesis[0]
        
        # Stub untuk attention, karena kita tidak menghitungnya di sini
        attn_stub = torch.zeros(best_seq.size(0), 1, src.size(0), device=src.device)
        return best_seq, attn_stub

    @torch.no_grad()
    def greedy_decode(self, src, max_len=50):
        """
        src: [Tsrc, B]
        returns: ys: [Tout, B]
        """
        ## PERBAIKAN: Gunakan size(1) untuk batch size
        B = src.size(1) 
        src_key_padding_mask = self.make_pad_mask(src)
        src_emb = self.pos_enc(self.src_embed(src) * math.sqrt(self.d_model))
        memory = self.transformer.encoder(src_emb, src_key_padding_mask=src_key_padding_mask)

        ## PERBAIKAN: Bangun `ys` dalam format [Time, Batch] dari awal
        ys = torch.full((1, B), self.bos_id, dtype=torch.long, device=src.device)

        for _ in range(1, max_len):
            tgt_emb = self.pos_enc(self.trg_embed(ys) * math.sqrt(self.d_model))
            ## PERBAIKAN: Gunakan size(0) untuk panjang sekuens
            tgt_mask = generate_square_subsequent_mask(ys.size(0), src.device)
            out = self.transformer.decoder(
                tgt=tgt_emb, memory=memory,
                tgt_mask=tgt_mask,
                tgt_key_padding_mask=self.make_pad_mask(ys),
                memory_key_padding_mask=src_key_padding_mask
            ) # out: [T_current, B, C]
            
            ## PERBAIKAN: Ambil output dari token terakhir (dimensi waktu)
            logits = self.generator(out[-1, :, :])  # out[-1, :, :] -> [B, C]
            next_tok = torch.argmax(logits, dim=-1) # [B]

            ## PERBAIKAN: Concat di dimensi waktu (dim=0)
            ys = torch.cat([ys, next_tok.unsqueeze(0)], dim=0)
            
            if (next_tok == self.eos_id).all():
                break
        
        # ys sudah dalam format [Tout, B], tidak perlu transpose
        attn_stub = torch.zeros(ys.size(0), B, src.size(0), device=src.device)
        return ys, attn_stub