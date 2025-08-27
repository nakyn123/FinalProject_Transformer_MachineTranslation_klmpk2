import torch.nn as nn
import torch

# Encoder dengan Bahdanau Attention
# Encoder membaca urutan input (bahasa sumber) dan menghasilkan representasi konteks
class BahdanauEncoder(nn.Module):
	def __init__(self, input_dim, embedding_dim, encoder_hidden_dim, decoder_hidden_dim, dropout_p):

		super().__init__()
		self.input_dim = input_dim
		self.embedding_dim = embedding_dim

		self.encoder_hidden_dim = encoder_hidden_dim
		self.decoder_hidden_dim = decoder_hidden_dim

		# Dropout agar model tidak overfitting
		self.dropout_p = dropout_p

		self.embedding = nn.Embedding(input_dim, embedding_dim)

		self.gru = nn.GRU(embedding_dim, encoder_hidden_dim, bidirectional=True)
		self.linear = nn.Linear(encoder_hidden_dim * 2, decoder_hidden_dim)

		self.dropout = nn.Dropout(dropout_p)

	def forward(self, x):
		x = self.embedding(x)
		embedded = self.dropout(x)

		outputs, hidden = self.gru(embedded)
		x = torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1)
		x = self.linear(x)

		hidden = torch.tanh(x)

		# outputs → semua hidden state untuk setiap langkah (dipakai attention)
		# hidden → state ringkasan untuk inisialisasi decoder
		return outputs, hidden