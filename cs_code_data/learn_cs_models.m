%Tune threshold and learn a SVM to combine visual and text
function learn_cs_models(val,w2v,abstract_scenes,PRS_model)

path_to_liblinear='./';
addpath(genpath(path_to_liblinear));
addpath(genpath('utils'));

nval=size(val.score,1);

val_P=val.P;
val_R=val.R;
val_S=val.S;

%index and get word2vec embeddings for val PRS
val_R_unique_label=unique(val_R);
val_R_embedding=cell(length(val_R_unique_label),1);
[~ , val_R_id]=ismember(val_R,val_R_unique_label);
for i=1:length(val_R_unique_label)
	val_R_embedding{i}=embed_str(val_R_unique_label{i},w2v.tokens,w2v.fv);
end
val_R_embedding=cell2mat(val_R_embedding);

val_P_unique_label=unique(val_P);
val_P_embedding=cell(length(val_P_unique_label),1);
[~ , val_P_id]=ismember(val_P,val_P_unique_label);
for i=1:length(val_P_unique_label)
	val_P_embedding{i}=embed_str(val_P_unique_label{i},w2v.tokens,w2v.fv);
end
val_P_embedding=cell2mat(val_P_embedding);

val_S_unique_label=unique(val_S);
val_S_embedding=cell(length(val_S_unique_label),1);
[~ , val_S_id]=ismember(val_S,val_S_unique_label);
for i=1:length(val_S_unique_label)
	val_S_embedding{i}=embed_str(val_S_unique_label{i},w2v.tokens,w2v.fv);
end
val_S_embedding=cell2mat(val_S_embedding);

%convert human scores to labels
val_score=double(val.score);
val_label=val_score>0;

%Visual model
%compute x'A for each x
R_score_test_embed=abstract_scenes.fv*PRS_model.R_A;
P_score_test_embed=abstract_scenes.fv*PRS_model.P_A;
S_score_test_embed=abstract_scenes.fv*PRS_model.S_A;

%compute x'Ay for each x and y in val
val_R_unique_score_embed=R_score_test_embed*val_R_embedding';
val_P_unique_score_embed=P_score_test_embed*val_P_embedding';
val_S_unique_score_embed=S_score_test_embed*val_S_embedding';

%Tune threshold until prec_val is maximized. Visual threshold usually around 0~1
thresholds=-1:0.1:2;
nthresholds=length(thresholds);
precs_val=zeros(nthresholds,1);
for t=1:nthresholds
	threshold=thresholds(t);
	visual_val_score=zeros(nval,1);
	%compute the score for each val instance
	for i=1:nval
		visual_val_score(i,1)=mean(max(val_P_unique_score_embed(:,val_P_id(i))+val_R_unique_score_embed(:,val_R_id(i))+val_S_unique_score_embed(:,val_S_id(i))-threshold,0),1);
	end

	%run evaluation multiple times to eliminate randomness in AP computation. 
	prec_acc_val=0;
	for i=1:1000
		[prec_val,base]=precision(visual_val_score,val_label);
		prec_acc_val=prec_acc_val+prec_val;
	end

	precs_val(t)=prec_acc_val/1000;
end
[visual_val_prec,ind]=max(precs_val);
threshold_visual=thresholds(ind);
visual_val_score=zeros(nval,1);
%compute val scores using optimal threshold
for i=1:nval
	visual_val_score(i,1)=mean(max(val_P_unique_score_embed(:,val_P_id(i))+val_R_unique_score_embed(:,val_R_id(i))+val_S_unique_score_embed(:,val_S_id(i))-threshold_visual,0),1);
end
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


%compute cosine similarity-1 for val
val_R_unique_score_embed_text=-pdist2(R_embedding(R_id,:),val_R_embedding,'cosine');
val_P_unique_score_embed_text=-pdist2(P_embedding(P_id,:),val_P_embedding,'cosine');
val_S_unique_score_embed_text=-pdist2(S_embedding(S_id,:),val_S_embedding,'cosine');


%for unseen words, the cosine similarity is nan. Set them to 0.
val_R_unique_score_embed_text(isnan(val_R_unique_score_embed_text(:)))=-1;
val_P_unique_score_embed_text(isnan(val_P_unique_score_embed_text(:)))=-1;
val_S_unique_score_embed_text(isnan(val_S_unique_score_embed_text(:)))=-1;


%Tune threshold until prec_val is maximized. Text threshold usually around -2~1
thresholds=-2:0.1:1;
nthresholds=length(thresholds);
precs_val=zeros(nthresholds,1);
for t=1:nthresholds
	threshold=thresholds(t);
	text_val_score=zeros(nval,1);
	for i=1:nval
		text_val_score(i,1)=mean(max(val_P_unique_score_embed_text(:,val_P_id(i))+val_R_unique_score_embed_text(:,val_R_id(i))+val_S_unique_score_embed_text(:,val_S_id(i))-threshold,0),1);
	end

	%run evaluation multiple times to eliminate randomness in AP computation. 
	prec_acc_val=0;
	for i=1:1000
		[prec_val,base]=precision(text_val_score,val_label);
		prec_acc_val=prec_acc_val+prec_val;
	end

	precs_val(t)=prec_acc_val/1000;
end
[text_val_prec,ind]=max(precs_val);
threshold_text=thresholds(ind);
%compute val scores using optimal threshold
text_val_score=zeros(nval,1);
for i=1:nval
	text_val_score(i,1)=mean(max(val_P_unique_score_embed_text(:,val_P_id(i))+val_R_unique_score_embed_text(:,val_R_id(i))+val_S_unique_score_embed_text(:,val_S_id(i))-threshold_text,0),1);
end



%Combining Text+Visual
%TODO: Choose your feature combination
[prec_val,base]=precision(visual_val_score,val_label)
[prec_val,base]=precision(text_val_score,val_label)
[prec_val,base]=precision(visual_val_score+text_val_score,val_label)

hybrid_feat_val=[text_val_score visual_val_score];

%TODO: Tune C till optimal. Typically C doesn't matter that much.
c=10000;
%crossval
[hybrid_model_test hybrid_model_crossval hybrid_acc_crossval hybrid_random_crossval]=perclass(val_label*2-1,sparse(hybrid_feat_val),c,5);
hybrid_model_test=hybrid_model_test{1};

save('model/cs_models.mat','threshold_visual','threshold_text','hybrid_model_test');