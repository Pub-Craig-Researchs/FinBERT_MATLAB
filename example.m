load FinBERT.mat

testStr =["there is a shortage of capital, and we need extra financing"];
tokens = FinBERT_encode(testStr,"vocab.txt");
[senti,score] = FinBERT_prediction(fin_bert,tokens);