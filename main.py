import unicodedata
from collections import Counter
from pathlib import Path
import argparse
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction #nltk ini untuk nilai BLEU nya
import json
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
import math
import matplotlib.pyplot as plt
import csv
import sentencepiece as spm #ini untuk tokenisasi subword

from util import * 
from encoder import BahdanauEncoder
from decoder import BahdanauDecoder
from attention import BahdanauAttentionQKV
from seq2seq import BahdanauSeq2Seq
from transformer import TransformerNMT 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLIP = 1.0  # clip grad norm


# ---- Noam/Warmup scheduler (tempel di main.py) ----
class NoamWarmup: #Learning rate naik perlahan saat warmup steps (biar stabil), lalu menurun sesuai step.
    """
    Scheduler gaya Noam: lr = d_model^{-0.5} * min(step^{-0.5}, step * warmup^{-1.5})
    """
    def __init__(self, optimizer, d_model=256, warmup_steps=4000, factor=1.0):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.factor = factor
        self.d_model = d_model
        self._step = 0

    def step(self):
        self._step += 1
        lr = self.factor * (self.d_model ** -0.5) * min(
            self._step ** -0.5, self._step * (self.warmup_steps ** -1.5)
        )
        for pg in self.optimizer.param_groups:
            pg['lr'] = lr
        return lr

    @property
    def step_num(self):
        return self._step

parser = argparse.ArgumentParser()
# Parameter ini mengatur dataset yang dipakai, ukuran batch, jumlah epoch pelatihan, laju belajar, serta tingkat dropout untuk regularisasi.
parser.add_argument('--data_path', type=str, default='data/ind.txt', help='Path to txt data')
parser.add_argument('--epochs', type=int, default=20, help='Number of training epochs')
parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
parser.add_argument('--tf', type=float, default=0.5, help='Teacher Forcing')
parser.add_argument('--dropout', type=float, default=0.15, help='dropout')
parser.add_argument('--num_classes', type=int, default=10, help='Number of classes')
parser.add_argument('--max_vocab', type=int, default=None)
parser.add_argument('--target_lang', type=str, default='ID', help='Bahasa tujuan')
parser.add_argument('--checkpoint', type=str, default='cat_dog_checkpoint.pth', help='Path to save model checkpoint')
# Menentukan cara memecah teks, apakah berbasis kata (word-level) atau sub-kata (sentencepiece).
parser.add_argument('--tokenizer', type=str, default='word', choices=['word','sp'],
                    help='Tokenisasi: word-level atau subword (sentencepiece)')
parser.add_argument('--sp_src_model', type=str, default=None, help='Path SentencePiece model source')
parser.add_argument('--sp_trg_model', type=str, default=None, help='Path SentencePiece model target')
# Memilih arsitektur model yang digunakan, apakah RNN dengan perhatian Bahdanau atau Transformer.
parser.add_argument('--model', type=str, default='rnn', choices=['rnn','transformer'],
                    help='Pilih arsitektur: rnn (Bahdanau) atau transformer')  
# Hyperparameter Transformer
# Mengatur dimensi embedding, jumlah kepala perhatian, ukuran feed-forward, dan parameter inti Transformer lainnya.
parser.add_argument('--d_model', type=int, default=256)      
parser.add_argument('--nhead', type=int, default=8)         
parser.add_argument('--enc_layers', type=int, default=4)     
parser.add_argument('--dec_layers', type=int, default=4)       
parser.add_argument('--ff_dim', type=int, default=1024)        
parser.add_argument('--pe_dropout', type=float, default=0.1)
# Label smoothing membantu mencegah overconfidence, sedangkan warmup steps mengatur laju belajar secara bertahap di awal pelatihan.    
parser.add_argument('--label_smoothing', type=float, default=0.0)  
parser.add_argument('--warmup_steps', type=int, default=0)      
parser.add_argument('--exp_name', type=str, default='exp')      
# === PERUBAHAN 1: Tambahkan argumen beam_size ===
# Menentukan jumlah jalur pencarian saat decoding untuk menghasilkan terjemahan yang lebih optimal.
parser.add_argument('--beam_size', type=int, default=3, help='Beam size untuk contoh inferensi di akhir training.')      
args = parser.parse_args()


SPECIALS = ["<pad>", "<bos>", "<eos>", "<unk>"]
PAD, BOS, EOS, UNK = range(4)
# <pad> : Token ini dipakai untuk meratakan panjang kalimat dalam batch, jadi model bisa memprosesnya sekaligus.
# <bos> : Token khusus yang selalu diletakkan di awal kalimat agar model tahu kapan mulai membaca/menulis.
# <eos> : Token penutup yang menunjukkan akhir sebuah kalimat, supaya model berhenti menghasilkan output.
# <unk> : Token pengganti untuk kata yang tidak ada di kosakata model, jadi model tetap bisa memproses teksnya.



# Kita ambil semua kata unik dari dataset, lalu tiap kata diberi nomor indeks (misalnya "saya" → 1, "makan" → 2) 
# supaya teks bisa diubah jadi angka untuk input ke model.
def build_vocab(token_lists, min_freq=1, max_size=None, specials=["<pad>", "<bos>", "<eos>", "<unk>"]):
    counter = Counter()
    for toks in token_lists:
        counter.update(toks)

    filtered = [(w, c) for w, c in counter.items() if c >= min_freq]
    filtered.sort(key=lambda x: (-x[1], x[0]))
    if max_size is not None:
        filtered = filtered[:max(0, max_size - len(specials))]
    vocab = {sp: i for i, sp in enumerate(specials)}
    for w, _ in filtered:
        if w not in vocab:
            vocab[w] = len(vocab)
    itos = {i: w for w, i in vocab.items()}
    return vocab, itos


class NMTDataset(Dataset):
    def __init__(self, pairs, src_vocab, trg_vocab):
        self.src_vocab = src_vocab
        self.trg_vocab = trg_vocab
        self.data = [(to_ids(src, src_vocab), to_ids(trg, trg_vocab)) for src, trg in pairs]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        src_ids, trg_ids = self.data[idx]
        return torch.tensor(src_ids, dtype=torch.long), torch.tensor(trg_ids, dtype=torch.long)


data_file = Path(args.data_path)
id_itos = None
sp_trg = None

if args.tokenizer == "word":
    pairs = load_pairs(data_file, max_len=20, max_pairs=None)
    train_pairs, val_pairs, test_pairs = split_pairs(pairs, 0.8, 0.1)

    en_vocab, en_itos = build_vocab([src for src, _ in train_pairs], max_size=args.max_vocab)
    id_vocab, id_itos = build_vocab([tgt for _, tgt in train_pairs], max_size=args.max_vocab)

    with open("en_vocab.json","w") as f: json.dump(en_vocab,f)
    with open("id_vocab.json","w") as f: json.dump(id_vocab,f)

    print(f"EN vocab size: {len(en_vocab)} | ID vocab size: {len(id_vocab)}")

    class NMTDataset(Dataset):
        def __init__(self, pairs, src_vocab, trg_vocab):
            self.data = [(to_ids(src, src_vocab), to_ids(trg, trg_vocab)) for src, trg in pairs]
        def __len__(self): return len(self.data)
        def __getitem__(self, idx):
            src_ids, trg_ids = self.data[idx]
            return torch.tensor(src_ids), torch.tensor(trg_ids)

    train_ds = NMTDataset(train_pairs, en_vocab, id_vocab)
    val_ds   = NMTDataset(val_pairs,   en_vocab, id_vocab)
    test_ds  = NMTDataset(test_pairs,  en_vocab, id_vocab)
    input_dim, output_dim = len(en_vocab), len(id_vocab)

else:  # sentencepiece
    sp_src = spm.SentencePieceProcessor()
    sp_src.Load(args.sp_src_model)
    sp_trg = spm.SentencePieceProcessor()
    sp_trg.Load(args.sp_trg_model)
    pairs = load_pairs_sp(data_file, sp_src, sp_trg, max_len=100)
    train_pairs, val_pairs, test_pairs = split_pairs(pairs, 0.8, 0.1)

    class NMTDatasetSP(Dataset):
        def __init__(self, pairs): self.data = pairs
        def __len__(self): return len(self.data)
        def __getitem__(self, idx):
            src_ids, trg_ids = self.data[idx]
            return torch.tensor(src_ids), torch.tensor(trg_ids)

    train_ds = NMTDatasetSP(train_pairs)
    val_ds   = NMTDatasetSP(val_pairs)
    test_ds  = NMTDatasetSP(test_pairs)
    input_dim  = sp_src.get_piece_size()
    output_dim = sp_trg.get_piece_size()


BATCH_SIZE = args.batch_size
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_batch)
val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_batch)
test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_batch)

# ==== konstruksi model ====
if args.model == 'rnn':

# ada encoder (ubah kalimat sumber jadi representasi)
# attention Bahdanau (pilih bagian penting dari kalimat)
# dan decoder (hasilin kalimat terjemahan).

    ENCODER_HIDDEN_SIZE = 512
    DECODER_HIDDEN_SIZE = 256
    ENCODER_EMBEDDING_DIM = 256
    DECODER_EMBEDDING_DIM = 256
    encoder = BahdanauEncoder(input_dim, ENCODER_EMBEDDING_DIM,
                              ENCODER_HIDDEN_SIZE, DECODER_HIDDEN_SIZE, dropout_p=args.dropout)
    attn = BahdanauAttentionQKV(
        hidden_size=DECODER_HIDDEN_SIZE, query_size=DECODER_HIDDEN_SIZE,
        key_size=2 * ENCODER_HIDDEN_SIZE, dropout_p=0.0
    )
    decoder = BahdanauDecoder(output_dim, DECODER_EMBEDDING_DIM,
                              ENCODER_HIDDEN_SIZE, DECODER_HIDDEN_SIZE,
                              attn, dropout_p=args.dropout)
    seq2seq = BahdanauSeq2Seq(encoder, decoder, device,
                              pad_id=PAD, bos_id=BOS, eos_id=EOS).to(device)
    is_transformer = False

# model lebih modern yang pakai multi-head attention (fokus ke banyak bagian kalimat sekaligus)
# plus positional encoding (supaya tahu urutan kata).
else:  # transformer
    seq2seq = TransformerNMT(
        src_vocab_size=input_dim, trg_vocab_size=output_dim,
        d_model=args.d_model, nhead=args.nhead,
        num_encoder_layers=args.enc_layers, num_decoder_layers=args.dec_layers,
        dim_feedforward=args.ff_dim, dropout=args.pe_dropout,
        pad_id=PAD, bos_id=BOS, eos_id=EOS, device=device
    ).to(device)
    is_transformer = True

# ========= Criterion + Optimizer =========

# dipakai untuk mengukur seberapa beda prediksi model dengan label asli; kalau Transformer bisa ditambah label smoothing supaya model tidak terlalu yakin pada 1 jawaban.
criterion = nn.CrossEntropyLoss(ignore_index=PAD, label_smoothing=args.label_smoothing if is_transformer else 0.0)
# algoritma buat ngatur bobot model selama training biar loss makin kecil.
optimizer = torch.optim.Adam(seq2seq.parameters(), lr=args.lr)
# khusus Transformer, atur learning rate mulai kecil → naik → turun lagi, supaya training lebih stabil.
scheduler = NoamWarmup(optimizer, d_model=args.d_model, warmup_steps=args.warmup_steps) if args.warmup_steps > 0 and is_transformer else None

def epoch_run(model, loader, train=True, teacher_forcing=0.5):
    model.train() if train else model.eval()
    total_loss, total_tokens = 0.0, 0
    with torch.set_grad_enabled(train):
        for src, trg in tqdm(loader):
            src, trg = src.to(device), trg.to(device)
            outputs, _att = model(src, trg, teacher_forcing_ratio=(teacher_forcing if train else 0.0))
            output_for_loss = outputs[:-1, :, :].contiguous()
            target_for_loss = trg[1:, :].contiguous()
            logits = output_for_loss.view(-1, outputs.size(-1))
            target = target_for_loss.view(-1)
            loss = criterion(logits, target)
            if train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP)
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()
            n_tokens = (target != PAD).sum().item()
            total_loss += loss.item() * n_tokens
            total_tokens += n_tokens
    avg_loss = total_loss / max(1, total_tokens)
    ppl = math.exp(avg_loss) if avg_loss < 20 else float("inf")
    return avg_loss, ppl

# (Fungsi plot_curves tidak perlu diubah, jadi saya hapus dari sini agar lebih ringkas)
def plot_curves(history, save_prefix="run", fontsize=14):
    # ... (kode Anda sama persis)
    pass

# -----------------------
# Train loop
# -----------------------
history = { "train_loss": [], "val_loss": [], "train_ppl": [], "val_ppl": [], "val_bleu": [] }
best_val = float("inf")
print(f"Running on: {device} | Model: {args.model} | Tokenizer: {args.tokenizer}")

for epoch in range(1, args.epochs + 1):
    tf = max(0.3, 0.7 - 0.04 * (epoch - 1)) if not is_transformer else 0.0
    train_loss, train_ppl = epoch_run(seq2seq, train_loader, train=True,  teacher_forcing=tf)
    val_loss,   val_ppl   = epoch_run(seq2seq, val_loader,   train=False, teacher_forcing=0.0)
    val_bleu = evaluate_sacrebleu(seq2seq, val_loader, trg_itos=id_itos, sp_trg=sp_trg)
    history["train_loss"].append(train_loss); history["val_loss"].append(val_loss)
    history["train_ppl"].append(train_ppl); history["val_ppl"].append(val_ppl)
    history["val_bleu"].append(val_bleu)
    print(f"Epoch {epoch:02d} | TF={tf:.2f} | Train Loss {train_loss:.4f} PPL {train_ppl:.2f} | Val Loss {val_loss:.4f} PPL {val_ppl:.2f} | Val BLEU {val_bleu:.2f}")
    if val_loss < best_val:
        best_val = val_loss
        torch.save(seq2seq.state_dict(), args.checkpoint)
        print(f"✓ Saved best to {args.checkpoint}")

# -------------------------------
# Save history CSV
# -------------------------------
hist_csv = f"{args.exp_name}_history.csv"
with open(hist_csv, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["epoch","train_loss","val_loss","train_ppl","val_ppl","val_bleu"])
    for i in range(len(history["train_loss"])):
        w.writerow([i+1, history["train_loss"][i], history["val_loss"][i], history["train_ppl"][i],  history["val_ppl"][i], history["val_bleu"][i]])
print(f"History saved to {hist_csv}")

# -------------------------------
# Evaluate on test (best ckpt)
# -------------------------------

# bagian ini dipakai untuk memuat bobot model terbaik yang sudah disimpan saat training.
seq2seq.load_state_dict(torch.load(args.checkpoint, map_location=device))

# menjalankan model pada data uji (test set) untuk menghitung hasil akhirnya, misalnya test loss dan perplexity.
test_loss, test_ppl = epoch_run(seq2seq, test_loader, train=False, teacher_forcing=0.0)

# Helper function
def clean_sp_ids(ids_list, eos_id):
    try:
        eos_index = ids_list.index(eos_id)
        return ids_list[1:eos_index]
    except ValueError:
        return ids_list[1:]

# ======================================================================
# === PERBAIKAN 2: Blok ini adalah versi yang sudah benar dan rapi ===
# ======================================================================

# Tampilkan beberapa contoh terjemahan + hitung BLEU
print("\n" + "="*20 + " Contoh Terjemahan " + "="*20)
seq2seq.eval()
with torch.no_grad():
    n_show = 5
    shown = 0
    
    # Buat DataLoader BARU khusus untuk sampling dengan batch_size=1
    # Ini penting agar beam search bisa bekerja pada satu kalimat.
    test_loader_sample = DataLoader(test_ds, batch_size=1, shuffle=True, collate_fn=collate_batch)

    # Loop ini akan mengambil SATU sampel (src, trg) setiap kali iterasi
    for src, trg in test_loader_sample:
        src = src.to(device)  # Shape: [T, 1]
        trg = trg.to(device)  # Shape: [T, 1]
        
        # Panggil fungsi decoding yang sesuai (beam search atau greedy)
        if args.model == 'transformer' and args.beam_size > 1:
            ys, _ = seq2seq.beam_search_decode(src, beam_size=args.beam_size, max_len=40)
        else:
            ys, _ = seq2seq.greedy_decode(src, max_len=40)

        # Karena batch size adalah 1, kita tidak perlu loop 'for b in range(B)'.
        # Cukup proses sampel di indeks 0.
        b = 0
        
        if args.tokenizer == "sp":
            pred_ids_list = ys[:, b].tolist()
            src_ids_list  = src[:, b].tolist() 
            trg_ids_list  = trg[:, b].tolist()

            cleaned_pred_ids = clean_sp_ids(pred_ids_list, EOS)
            cleaned_src_ids  = clean_sp_ids(src_ids_list, EOS)
            cleaned_trg_ids  = clean_sp_ids(trg_ids_list, EOS)

            pred_txt = sp_trg.decode(cleaned_pred_ids)
            src_txt  = sp_src.decode(cleaned_src_ids)
            trg_txt  = sp_trg.decode(cleaned_trg_ids)
        else: # word tokenizer
            pred_txt = decode_ids(ys[:, b], id_itos)
            src_txt  = decode_ids(src[:, b], en_itos)
            trg_txt  = decode_ids(trg[:, b], id_itos)

        print("-" * 60)
        print(f"SRC : {src_txt}")
        print(f"TRG : {trg_txt}")
        print(f"PRED: {pred_txt}")
        
        shown += 1
        if shown >= n_show:
            break

# Hitung skor BLEU untuk seluruh data test
test_bleu = evaluate_sacrebleu(
    seq2seq, 
    test_loader, 
    trg_itos=id_itos if args.tokenizer == 'word' else None,
    sp_trg=sp_trg if args.tokenizer == 'sp' else None
)
print("\n" + "="*25 + " Hasil Akhir " + "="*25)
print(f"TEST | Loss {test_loss:.4f} | PPL {test_ppl:.2f} | SacreBLEU {test_bleu:.2f}")