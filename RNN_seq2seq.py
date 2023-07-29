import torch
import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.rnn = nn.RNN(hidden_size, hidden_size, batch_first=True)

    def forward(self, input_seq):
        embedded = self.embedding(input_seq)
        _, hidden = self.rnn(embedded)
        return hidden

class Decoder(nn.Module):
    def __init__(self, output_size, hidden_size):
        super(Decoder, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(output_size, hidden_size)
        self.rnn = nn.RNN(hidden_size, hidden_size, batch_first=True)
        self.output_layer = nn.Linear(hidden_size, output_size)

    def forward(self, target_seq, hidden):
        embedded = self.embedding(target_seq)
        output, _ = self.rnn(embedded, hidden)
        output = self.output_layer(output)
        return output

class Seq2Seq(nn.Module):
    def __init__(self, input_size, output_size, hidden_size):
        super(Seq2Seq, self).__init__()
        self.encoder = Encoder(input_size, hidden_size)
        self.decoder = Decoder(output_size, hidden_size)

    def forward(self, input_seq, target_seq):
        encoder_hidden = self.encoder(input_seq)
        decoder_output = self.decoder(target_seq, encoder_hidden)
        return decoder_output

# 단어 사전
input_vocab = {"hello": 0, "world": 1, "good": 2, "morning": 3}
output_vocab = {0: "안녕", 1: "세계", 2: "좋은", 3: "아침"}

# 모델 초기화
input_size = len(input_vocab)
output_size = len(output_vocab)
hidden_size = 64
model = Seq2Seq(input_size, output_size, hidden_size)

# 입력과 출력 데이터 생성 (단어 인덱스로 표현)
input_seq = torch.tensor([[0, 1, 2, 3]])  # 입력 문장의 인덱스 시퀀스
target_seq = torch.tensor([[0, 1, 2, 3]])  # 출력 문장의 인덱스 시퀀스

# 모델에 입력하여 출력 얻기
output_seq = model(input_seq, target_seq)

# 출력 시퀀스를 단어로 변환하여 출력
output_words = [output_vocab[idx.item()] for idx in output_seq.argmax(dim=2).squeeze()]
print(output_words)
