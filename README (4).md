Proyek Machine Translation: RNN vs Transformer

Kelompok 2
Amisha Hersavina
Aisha Utwa Haura
Azmi Nur
Nasywa Kynda
Nur Aisyah Maharani
Program Studi: Teknik Informatika / Semester 5

Tujuan dari penelitian ini adalah mengevaluasi kinerja model Transformer yang dikombinasikan dengan tokenisasi SentencePiece untuk tugas penerjemahan bahasa. Selain itu, berfokus pada bagaimana arsitektur Transformer yang unggul dalam menangani dependensi jarak jauh dapat bersinergi dengan tokenisasi SentencePiece yang efektif mengatasi masalah kosakata, demi menghasilkan terjemahan yang lebih akurat dan efisien.

.
├── checkpoints/              # Folder untuk menyimpan model (.pt) yang sudah dilatih
├── data/
│   └── ind.txt               # File dataset bilingual mentah (en-id)
│
├── main.py                   # Skrip utama untuk menjalankan proses training.
├── eval.py                   # Skrip untuk mengevaluasi model pada test set.
├── prepare_data.py           # Skrip untuk memisahkan data mentah menjadi file per bahasa.
├── sp_train.py               # Skrip untuk melatih model tokenizer SentencePiece.
│
├── encoder.py                # Implementasi modul Encoder untuk model RNN.
├── decoder.py                # Implementasi modul Decoder untuk model RNN.
├── attention.py              # Implementasi mekanisme Attention (Bahdanau).
├── seq2seq.py                # Wrapper yang menggabungkan Encoder-Decoder RNN.
├── transformer.py            # Implementasi arsitektur Transformer.
│
├── util.py                   # Kumpulan fungsi bantuan (data loading, padding, dll).
├── analisis.py               # Skrip untuk analisis eksplorasi data (misal: panjang kalimat).
├── top_words.py              # Skrip untuk visualisasi kata-kata paling sering muncul.
├── heatmap.py                # Skrip untuk membuat visualisasi heatmap dari mekanisme atensi.
│
├── en_vocab.json             # File vocabulary hasil tokenisasi word-level untuk B. Inggris.
├── id_vocab.json             # File vocabulary hasil tokenisasi word-level untuk B. Indonesia.
├── spm_en.model / spm_id.model # File model SentencePiece yang sudah dilatih.
├── *.csv                     # File riwayat hasil training (loss, PPL, BLEU).
└── README.md                 # File panduan ini.

Eksperimen 1: Baseline RNN + Tokenizer Word-Level
python main.py --model rnn --tokenizer word --epochs 20 --lr 1e-4 --dropout 0.3 --exp_name rnn_word --checkpoint checkpoints/rnn_word.pt

Eksperimen 2: Baseline RNN + Tokenizer Subword (Studi Ablasi)
python main.py --model rnn --tokenizer sp --sp_src_model spm_en.model --sp_trg_model spm_id.model --epochs 20 --lr 5e-4 --dropout 0.2 --exp_name rnn_subword --checkpoint checkpoints/rnn_subword.pt

Eksperimen 3: Transformer + Tokenizer Subword
python main.py --model transformer --tokenizer sp --sp_src_model spm_en.model --sp_trg_model spm_id.model --epochs 20 --d_model 256 --nhead 8 --enc_layers 4 --dec_layers 4 --ff_dim 1024 --warmup_steps 4000 --label_smoothing 0.1 --exp_name transformer_subword --checkpoint checkpoints/transformer_subword.pt

Eksperimen 4: Transformer + Tokenizer Word-Level (Studi Ablasi)
python main.py --model transformer --tokenizer word --epochs 20 --d_model 256 --nhead 8 --enc_layers 4 --dec_layers 4 --ff_dim 1024 --warmup_steps 4000 --label_smoothing 0.1 --exp_name transformer_word --checkpoint checkpoints/transformer_word.pt