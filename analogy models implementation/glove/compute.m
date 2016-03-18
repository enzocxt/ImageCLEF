addpath('./eval');
vocab_file = 'words.txt';
vectors_file = 'vectors.txt';

fid = fopen(vocab_file, 'r');
words = textscan(fid, '%s');
fclose(fid);
words = words{1};
vocab_size = length(words);
global wordMap
wordMap = containers.Map(words(1:vocab_size),1:vocab_size);

W = dlmread(vectors_file, ' ');
W = bsxfun(@rdivide,W,sqrt(sum(W.*W,2))); %normalize vectors before evaluation
evaluate_vectors(W);
%exit

