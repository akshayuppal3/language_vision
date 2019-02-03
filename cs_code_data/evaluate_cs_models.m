%Evaluate models
function evaluate_cs_models(test,w2v,abstract_scenes,PRS_model,cs_models)

path_to_liblinear='./';
addpath(genpath(path_to_liblinear));
addpath(genpath('utils'));

ntest=size(test.score,1);

test_P=test.P;
test_R=test.R;
test_S=test.S;


test_R_unique_label=unique(test_R);
test_R_embedding=cell(length(test_R_unique_label),1);
[~ , test_R_id]=ismember(test_R,test_R_unique_label);
for i=1:length(test_R_unique_label)
	test_R_embedding{i}=embed_str(test_R_unique_label{i},w2v.tokens,w2v.fv);
end
test_R_embedding=cell2mat(test_R_embedding);

test_P_unique_label=unique(test_P);
test_P_embedding=cell(length(test_P_unique_label),1);
[~ , test_P_id]=ismember(test_P,test_P_unique_label);
for i=1:length(test_P_unique_label)
	test_P_embedding{i}=embed_str(test_P_unique_label{i},w2v.tokens,w2v.fv);
end
test_P_embedding=cell2mat(test_P_embedding);

test_S_unique_label=unique(test_S);
test_S_embedding=cell(length(test_S_unique_label),1);
[~ , test_S_id]=ismember(test_S,test_S_unique_label);
for i=1:length(test_S_unique_label)
	test_S_embedding{i}=embed_str(test_S_unique_label{i},w2v.tokens,w2v.fv);
end
test_S_embedding=cell2mat(test_S_embedding);

test_score=double(test.score);
test_label=test_score>0;


%Visual model
%compute x'A for each x
R_score_test_embed=abstract_scenes.fv*PRS_model.R_A;
P_score_test_embed=abstract_scenes.fv*PRS_model.P_A;
S_score_test_embed=abstract_scenes.fv*PRS_model.S_A;


%compute x'Ay for each x and y in test
test_R_unique_score_embed=R_score_test_embed*test_R_embedding';
test_P_unique_score_embed=P_score_test_embed*test_P_embedding';
test_S_unique_score_embed=S_score_test_embed*test_S_embedding';

visual_test_score=zeros(ntest,1);
for i=1:ntest
	visual_test_score(i,1)=mean(max(test_P_unique_score_embed(:,test_P_id(i))+test_R_unique_score_embed(:,test_R_id(i))+test_S_unique_score_embed(:,test_S_id(i))-cs_models.threshold_visual,0),1);
end
visual_test_prec=0;
for i=1:1000
	[prec_test,base]=precision(visual_test_score,test_label);
	visual_test_prec=visual_test_prec+prec_test;
end
visual_test_prec=visual_test_prec/1000;


disp('Visual Only');
disp(['AP: ',num2str(visual_test_prec)]);
disp(['Rank Corr: ',num2str(corr(visual_test_score, test_score, 'type', 'Spearman'))]);


%Text model
%Compute average word2vec for PRS in clipart data
%R
R_unique_label=unique(abstract_scenes.R);
[~ , R_id]=ismember(abstract_scenes.R,R_unique_label);
R_embedding=cell(length(R_unique_label),1);
for i=1:length(R_unique_label)
	R_embedding{i}=embed_str(R_unique_label{i},w2v.tokens,w2v.fv);
end
R_embedding=cell2mat(R_embedding);
%P
P_unique_label=unique(abstract_scenes.P);
[~ , P_id]=ismember(abstract_scenes.P,P_unique_label);
P_embedding=cell(length(P_unique_label),1);
for i=1:length(P_unique_label)
	P_embedding{i}=embed_str(P_unique_label{i},w2v.tokens,w2v.fv);
end
P_embedding=cell2mat(P_embedding);
%S
S_unique_label=unique(abstract_scenes.S);
[~ , S_id]=ismember(abstract_scenes.S,S_unique_label);
S_embedding=cell(length(S_unique_label),1);
for i=1:length(S_unique_label)
	S_embedding{i}=embed_str(S_unique_label{i},w2v.tokens,w2v.fv);
end
S_embedding=cell2mat(S_embedding);

%compute cosine similarity-1 for test.
%disable warnings to make it look nicer
warning('off');
test_R_unique_score_embed_text=-pdist2(R_embedding(R_id,:),test_R_embedding,'cosine');
test_P_unique_score_embed_text=-pdist2(P_embedding(P_id,:),test_P_embedding,'cosine');
test_S_unique_score_embed_text=-pdist2(S_embedding(S_id,:),test_S_embedding,'cosine');
warning('on');

test_R_unique_score_embed_text(isnan(test_R_unique_score_embed_text(:)))=-1;
test_P_unique_score_embed_text(isnan(test_P_unique_score_embed_text(:)))=-1;
test_S_unique_score_embed_text(isnan(test_S_unique_score_embed_text(:)))=-1;

text_test_score=zeros(ntest,1);
for i=1:ntest
	text_test_score(i,1)=mean(max(test_P_unique_score_embed_text(:,test_P_id(i))+test_R_unique_score_embed_text(:,test_R_id(i))+test_S_unique_score_embed_text(:,test_S_id(i))-cs_models.threshold_text,0),1);
end
text_test_prec=0;
for i=1:1000
	[prec,base]=precision(text_test_score,test_label);
	text_test_prec=text_test_prec+prec;
end
text_test_prec=text_test_prec/1000;

disp('Text Only');
disp(['AP: ',num2str(text_test_prec)]);
disp(['Rank Corr: ',num2str(corr(text_test_score, test_score, 'type', 'Spearman'))]);


%Visual+text model
hybrid_feat_test=[text_test_score visual_test_score];
[~,~,hybrid_test_score]=predict(zeros(size(test_label))*2-1,sparse(hybrid_feat_test),cs_models.hybrid_model_test,'-q');
hybrid_test_score=hybrid_test_score*cs_models.hybrid_model_test.Label(1);

hybrid_test_prec=0;
for i=1:1000
	[prec,base]=precision(hybrid_test_score,2*test_label-1);
	hybrid_test_prec=hybrid_test_prec+prec;
end
hybrid_test_prec=hybrid_test_prec/1000;

disp('Visual+Text');
disp(['AP: ',num2str(hybrid_test_prec)]);
disp(['Rank Corr: ',num2str(corr(hybrid_test_score, test_score, 'type', 'Spearman'))]);


