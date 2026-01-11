function tokens = FinBERT_encode(Str,enc_txt)
%FINBERT_ENCODE 此处显示有关此函数的摘要
%   此处显示详细说明
arguments (Input)
    Str
    enc_txt (1,1) string = "vocab.txt"
end

arguments (Output)
    tokens
end

vocab = readlines(enc_txt);
vocab = vocab(strlength(vocab) > 0);
enc = bertTokenizer(vocab);
tokens = encode(enc, Str);
tokens = gpuArray(dlarray(cell2mat(tokens)-1,"BC"));
end