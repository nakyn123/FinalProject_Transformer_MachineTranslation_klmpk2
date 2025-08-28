Proyek Machine Translation: Penerapan Arsitektur Transformer dengan
Tokenisasi SentencePiece menggunakan decoder Beam Search pada Penerjemahan Inggris-Indonesia

Kelompok 2 :
Amisha Hersavina
Aisha Utwa Haura
Azmi Nur
Nasywa Kynda
Nur Aisyah Maharani
Program Studi: Teknik Informatika / Semester 5

Tujuan utama dari penelitian ini adalah mengevaluasi dan menganalisis efektivitas arsitektur Transformer yang dikombinasikan dengan teknik tokenisasi SentencePiece untuk tugas penerjemahan mesin dari bahasa Inggris ke bahasa Indonesia. Kami berfokus pada bagaimana arsitektur Transformer, dengan mekanisme self-attention, mampu memahami konteks kata dalam kalimat secara menyeluruh, serta bagaimana tokenisasi SentencePiece secara efektif mengatasi masalah kosakata dengan memecah teks menjadi unit sub-kata. Melalui evaluasi yang menggunakan SacreBLEU Score, penelitian ini bertujuan untuk membuktikan bahwa integrasi kedua metode ini menghasilkan terjemahan yang lebih akurat, adaptif, dan alami, sehingga berpotensi menjadi dasar bagi pengembangan penerjemah modern yang lebih efisien dan mudah diimplementasikan.

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

run awal untuk data latih
python main.py --model transformer --tokenizer sp --sp_src_model spm_en.model --sp_trg_model spm_id.model --epochs 20 --d_model 256 --nhead 8 --enc_layers 4 --dec_layers 4 --ff_dim 1024 --warmup_steps 4000 --label_smoothing 0.1 --exp_name transformer_subword --checkpoint checkpoints/transformer_subword.pt

run untuk eval beam search
python eval.py --model transformer --tokenizer sp --data_path data/ind.txt --sp_src_model spm_en.model --sp_trg_model spm_id.model --checkpoint checkpoints/transformer_subword.pt