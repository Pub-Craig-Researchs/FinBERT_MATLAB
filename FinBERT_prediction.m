function [senti,score] = FinBERT_prediction(net,tokens)
% LABEL_0: neutral; LABEL_1: positive; LABEL_2: negative
% score = P(Pos.) - P(Neg.)
arguments (Input)
    net (1,1) dlnetwork
    tokens (:,:)
end

arguments (Output)
    senti string
    score gpuArray
end

mask = gpuArray(dlarray(ones(size(tokens,2),size(tokens,1)),"BC"));
logits = extractdata(net.predict(tokens,mask));
probabilities = softmax(logits');
score = probabilities(2,:) - probabilities(end,:);
[~,label] = max(probabilities,[],1);

senti = strings(length(label),1);
for i = 1:length(label)
    switch label(i)
        case 1
            senti(i) = "neutral";
        case 2
            senti(i) = "positive";
        case 3
            senti(i) = "negative";
    end
end