from util import *
from encoder import BahdanauEncoder
from decoder import BahdanauDecoder
from attention import BahdanauAttentionQKV
from seq2seq import BahdanauSeq2Seq
import json, torch, argparse
from torch.utils.data import DataLoader
from tqdm import tqdm
from pathlib import Path
import sacrebleu

# --- Parser Argumen ---
parser = argparse.ArgumentParser()
parser.add_argument('--data_path', type=str, default='data/ind.txt')
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--sp_src_model', type=str, default=None)
parser.add_argument('--sp_trg_model', type=str, default=None)
parser.add_argument('--tokenizer', type=str, default='word', choices=['word','sp'])
parser.add_argument('--checkpoint', type=str, default='bahdanau_best.pt')
parser.add_argument('--model', type=str, default='rnn', choices=['rnn','transformer'])
# PERBAIKAN: Argumen sudah benar, menggunakan nargs='+'
parser.add_argument('--beam_sizes', type=int, nargs='+', default=[1, 3, 5, 10], help='Daftar beam size yang akan dievaluasi.')
args = parser.parse_args()

# --- Setup Awal ---
PAD, BOS, EOS, UNK = 0, 1, 2, 3
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ===== Dataset & Vocab =====
data_file = Path(args.data_path)

if args.tokenizer == "word":
    with open("en_vocab.json") as f: en_vocab = {k:int(v) for k,v in json.load(f).items()}
    with open("id_vocab.json") as f: id_vocab = {k:int(v) for k,v in json.load(f).items()}
    en_itos = {v:k for k,v in en_vocab.items()}
    id_itos = {v:k for k,v in id_vocab.items()}

    pairs = load_pairs(data_file, max_len=20)
    _, _, test_pairs = split_pairs(pairs, 0.8, 0.1)
    test_ds = [(to_ids(src, en_vocab), to_ids(trg, id_vocab)) for src,trg in test_pairs]
    input_dim, output_dim = len(en_vocab), len(id_vocab)

else:  # sentencepiece
    sp_src = load_sentencepiece(args.sp_src_model)
    sp_trg = load_sentencepiece(args.sp_trg_model)
    pairs = load_pairs_sp(data_file, sp_src, sp_trg, max_len=100)
    _, _, test_pairs = split_pairs(pairs, 0.8, 0.1)
    test_ds = test_pairs
    # PERBAIKAN: Gunakan metode get_piece_size() yang benar
    input_dim  = sp_src.get_piece_size()
    output_dim = sp_trg.get_piece_size()

# Loader default, akan kita gunakan jika k=1 atau batch_size=1
test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_batch)

# ===== Model =====
if args.model == "rnn":
    encoder = BahdanauEncoder(input_dim, 256, 512, 256, dropout_p=0.3)
    attn = BahdanauAttentionQKV(256, 256, 1024)
    decoder = BahdanauDecoder(output_dim, 256, 512, 256, attn, dropout_p=0.3)
    seq2seq = BahdanauSeq2Seq(encoder, decoder, device, pad_id=PAD, bos_id=BOS, eos_id=EOS).to(device)
else:
    from transformer import TransformerNMT
    seq2seq = TransformerNMT(
        src_vocab_size=input_dim,
        trg_vocab_size=output_dim,
        d_model=256,
        nhead=8,
        num_encoder_layers=4,
        num_decoder_layers=4,
        dim_feedforward=1024,
        dropout=0.1,
        pad_id=PAD, bos_id=BOS, eos_id=EOS,
        device=device
    ).to(device)

seq2seq.load_state_dict(torch.load(args.checkpoint, map_location=device))
seq2seq.eval()

# ===== Evaluation =====
all_results = {}

# PERBAIKAN: Memulai loop untuk setiap beam size
for k in args.beam_sizes:
    print("-" * 50)
    print(f"🚀 Mengevaluasi dengan Beam Size = {k}...")

    # PERBAIKAN: Inisialisasi list di DALAM loop agar reset untuk setiap k
    references, hypotheses = [], []

    # PERBAIKAN: Seluruh blok evaluasi sekarang ada di DALAM loop
    with torch.no_grad():
        # PERBAIKAN: Logika untuk menentukan loader yang akan dipakai
        current_batch_size = args.batch_size
        if k > 1 and current_batch_size != 1:
            print(f"   -> Batch size diubah sementara menjadi 1 untuk beam search (k={k}).")
            loader = DataLoader(test_ds, batch_size=1, shuffle=False, collate_fn=collate_batch)
        else:
            loader = test_loader # Gunakan loader asli

        for src, trg in tqdm(loader, desc=f"Beam(k={k})"):
            src, trg = src.to(device), trg.to(device)
            
            # PERBAIKAN: Menggunakan variabel 'k' dari loop, bukan 'args.beam_size'
            if args.model == 'transformer' and k > 1:
                ys, _ = seq2seq.beam_search_decode(src, beam_size=k, max_len=40)
            else:
                # k=1 adalah sama dengan greedy decode
                ys, _ = seq2seq.greedy_decode(src, max_len=40)

            B = src.size(1)
            for b in range(B):
                if args.tokenizer == "sp":
                    # Kita sudah punya kalimat utuh dalam bentuk string, JANGAN di-split()
                    pred_str = sp_trg.decode_ids(ys[:,b].tolist())
                    ref_str  = sp_trg.decode_ids(trg[:,b].tolist())
                else:
                    # Fungsi decode_ids sudah menghasilkan string kalimat utuh
                    pred_str = decode_ids(ys[:,b], id_itos)
                    ref_str  = decode_ids(trg[:,b], id_itos)

                # PERBAIKAN: Masukkan string kalimat utuh, bukan list kata
                hypotheses.append(pred_str)
                references.append(ref_str)

    # PERBAIKAN: Kalkulasi SacreBLEU ada di DALAM loop
    # Kita tidak perlu NLTK BLEU karena SacreBLEU lebih standar
    sacre = sacrebleu.corpus_bleu(hypotheses, [references])
    
    # PERBAIKAN: Simpan hasil ke dictionary dan cetak
    all_results[k] = sacre.score
    print(f"✅ Hasil untuk Beam Size = {k} | SacreBLEU: {sacre.score:.2f}")

# PERBAIKAN: Mencetak tabel rangkuman di akhir, di LUAR loop
print("\n" + "="*50)
print("🏁 HASIL AKHIR EVALUASI 🏁")
print("="*50)
print(f"{'Beam Size (k)':<15} | {'SacreBLEU Score':<20}")
print("-" * 50)
for k, score in all_results.items():
    print(f"{k:<15} | {score:<20.2f}")
print("="*50)