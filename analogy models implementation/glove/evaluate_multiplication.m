function [BB] = evaluate_multiplication(W)

global wordMap

filenames = {'capital-common-countries' 'capital-world' 'currency' 'city-in-state' 'family' 'gram1-adjective-to-adverb' ...
    'gram2-opposite' 'gram3-comparative' 'gram4-superlative' 'gram5-present-participle' 'gram6-nationality-adjective' ...
    'gram7-past-tense' 'gram8-plural' 'gram9-plural-verbs'};
path = 'C:/Users/Babyburger/Desktop/glove/eval/question-data/';
%path = './eval/question-data/';

split_size = 100; %to avoid memory overflow, could be increased/decreased depending on system and vocab size

correct_sem = 0; %count correct semantic questions
correct_syn = 0; %count correct syntactic questions
correct_tot = 0; %count correct questions
count_sem = 0; %count all semantic questions
count_syn = 0; %count all syntactic questions
count_tot = 0; %count all questions
full_count = 0; %count all questions, including those with unknown words

if wordMap.isKey('<unk>')
    unkkey = wordMap('<unk>');
else
    unkkey = 0;
end

for j=1:length(filenames);

clear dist;

fid=fopen([path filenames{j} '.txt']);
temp=textscan(fid,'%s%s%s%s');
fclose(fid);
ind1 = cellfun(@WordLookup,temp{1}); %indices of first word in analogy
ind2 = cellfun(@WordLookup,temp{2}); %indices of second word in analogy
ind3 = cellfun(@WordLookup,temp{3}); %indices of third word in analogy
ind4 = cellfun(@WordLookup,temp{4}); %indices of answer word in analogy
full_count = full_count + length(ind1);
ind = (ind1 ~= unkkey) & (ind2 ~= unkkey) & (ind3 ~= unkkey) & (ind4 ~= unkkey); %only look at those questions which have no unknown words
ind1 = ind1(ind);
ind2 = ind2(ind);
ind3 = ind3(ind);
ind4 = ind4(ind);
disp([filenames{j} ':']);
mx = zeros(1,length(ind1));
num_iter = ceil(length(ind1)/split_size);
for jj=1:num_iter
range = (jj-1)*split_size+1:min(jj*split_size,length(ind1));
cresult = W * W(ind3(range),:)'; %cosine similarity if input W has been normalized
bresult = W * W(ind2(range),:)';
aresult = W * W(ind1(range),:)';
[m,n] = size(aresult);
[pp,~] = size(W(ind1(range),:));
dist = zeros(400000,pp);
%aresult = arrayfun(@(x) (x+1)/2, aresult);
%bresult = arrayfun(@(x) (x+1)/2, bresult);
%cresult = arrayfun(@(x) (x+1)/2, cresult);
%aresult = arrayfun(@(x) if(x==0) x = 0.001 end, aresult);

for nr=1:m
    for nc=1:n
        aresult(nr, nc) = (aresult(nr, nc)+1)/2;
        bresult(nr, nc) = (bresult(nr, nc)+1)/2;
        cresult(nr, nc) = (cresult(nr, nc)+1)/2;
        
        if(aresult == 0) 
            aresult = 0.001;
        end
        
        dist(nr, nc) = (cresult(nr, nc) * bresult(nr, nc))/aresult(nr, nc);
    end
end

dist
%dist = arrayfun(@(a,b,c) (c*b)/a, aresult, bresult, cresult)

for i=1:length(range)
dist(ind1(range(i)),i) = -Inf;
dist(ind2(range(i)),i) = -Inf;
dist(ind3(range(i)),i) = -Inf;
end
[~, mx(range)] = max(dist); %predicted word index
end

val = (ind4 == mx'); %correct predictions
count_tot = count_tot + length(ind1);
correct_tot = correct_tot + sum(val);
disp(['ACCURACY TOP1: ' num2str(mean(val)*100,'%-2.2f') '%  (' num2str(sum(val)) '/' num2str(length(val)) ')']);
if j < 6
    count_sem = count_sem + length(ind1);
    correct_sem = correct_sem + sum(val);
else
    count_syn = count_syn + length(ind1);
    correct_syn = correct_syn + sum(val);
end
    
disp(['Total accuracy: ' num2str(100*correct_tot/count_tot,'%-2.2f') '%   Semantic accuracy: ' num2str(100*correct_sem/count_sem,'%-2.2f') '%    Syntactic accuracy: ' num2str(100*correct_syn/count_syn,'%-2.2f') '%']);

end
disp('________________________________________________________________________________');
disp(['Questions seen/total: ' num2str(100*count_tot/full_count,'%-2.2f') '%  (' num2str(count_tot) '/' num2str(full_count) ')']);
disp(['Semantic Accuracy: ' num2str(100*correct_sem/count_sem,'%-2.2f') '%   (' num2str(correct_sem) '/' num2str(count_sem) ')']);
disp(['Syntactic Accuracy: ' num2str(100*correct_syn/count_syn,'%-2.2f') '%   (' num2str(correct_syn) '/' num2str(count_syn) ')']);
disp(['Total Accuracy: ' num2str(100*correct_tot/count_tot,'%-2.2f') '%   (' num2str(correct_tot) '/' num2str(count_tot) ')']);
BB = [100*correct_sem/count_sem 100*correct_syn/count_syn 100*correct_tot/count_tot];

end
