%Train a bilinear scoring function for abstract scenes using all scenes
%Quite slow, takes two hour or so.
function learn_PRS_models(abstract_scenes,w2v)
path_to_liblinear='./';
addpath(genpath(path_to_liblinear));
addpath(genpath('utils'));

%-----------------Preprocessing----------------
%Generate -1/1 labels for PRS *_label_01
%Compute average word2vec for PRS *_embedding
%R
nim=size(abstract_scenes.fv,1);
R_unique_label=unique(abstract_scenes.R);
[~ , R_id]=ismember(abstract_scenes.R,R_unique_label);
R_label_01=zeros(nim,length(R_unique_label));
for i=1:nim
	R_label_01(i,R_id(i))=1;
end
R_label_01=2*R_label_01-1;

R_embedding=cell(length(R_unique_label),1);
for i=1:length(R_unique_label)
	disp(num2str(i));
	R_embedding{i}=embed_str(R_unique_label{i},w2v.tokens,w2v.fv);
end
R_embedding=cell2mat(R_embedding);
%P
P_unique_label=unique(abstract_scenes.P);
[~ , P_id]=ismember(abstract_scenes.P,P_unique_label);
P_label_01=zeros(nim,length(P_unique_label));
for i=1:nim
	P_label_01(i,P_id(i))=1;
end
P_label_01=2*P_label_01-1;

P_embedding=cell(length(P_unique_label),1);
for i=1:length(P_unique_label)
	disp(num2str(i));
	P_embedding{i}=embed_str(P_unique_label{i},w2v.tokens,w2v.fv);
end
P_embedding=cell2mat(P_embedding);
%S
S_unique_label=unique(abstract_scenes.S);
[~ , S_id]=ismember(abstract_scenes.S,S_unique_label);
S_label_01=zeros(nim,length(S_unique_label));
for i=1:nim
	S_label_01(i,S_id(i))=1;
end
S_label_01=2*S_label_01-1;

S_embedding=cell(length(S_unique_label),1);
for i=1:length(S_unique_label)
	disp(num2str(i));
	S_embedding{i}=embed_str(S_unique_label{i},w2v.tokens,w2v.fv);
end
S_embedding=cell2mat(S_embedding);
%features
ndims=size(abstract_scenes.fv,2);
ndims_w2v=size(w2v.fv,2);

%-----------------Learn A_P, A_R and A_S----------------
%Number of folds for crossval
nfolds=5;
%Range of C values.
Cs=[0.0001 0.001 0.01 0.1];
%Number of random negative samples. There are too many negative clipart-word matchings.
nneg_samples=12780;
%Fixed seed
rng_seed=100;

[R_model_test_embed R_model_crossval_embed R_acc_crossval_embed R_random_crossval_embed]=train_embedding(R_label_01,R_embedding,abstract_scenes.fv,Cs,nfolds,nneg_samples,rng_seed)
[P_model_test_embed P_model_crossval_embed P_acc_crossval_embed P_random_crossval_embed]=train_embedding(P_label_01,P_embedding,abstract_scenes.fv,Cs,nfolds,nneg_samples,rng_seed)
[S_model_test_embed S_model_crossval_embed S_acc_crossval_embed S_random_crossval_embed]=train_embedding(S_label_01,S_embedding,abstract_scenes.fv,Cs,nfolds,nneg_samples,rng_seed)

[~,ind_R]=max(mean(R_acc_crossval_embed,2),[],1);
[~,ind_P]=max(mean(P_acc_crossval_embed,2),[],1);
[~,ind_S]=max(mean(S_acc_crossval_embed,2),[],1);
R_A=reshape(R_model_test_embed{ind_R}.w,[ndims,ndims_w2v]);
P_A=reshape(P_model_test_embed{ind_P}.w,[ndims,ndims_w2v]);
S_A=reshape(S_model_test_embed{ind_S}.w,[ndims,ndims_w2v]);

%Save the matrix A_P, A_R, A_S for the bilinear part. The biases are constant because we use liblinear.
%Also save the index for the C ind_P, ind_R and ind_S to check if they are near the border of C range.
save('model/clipart_PRS.mat','P_A','R_A','S_A','ind_P','ind_R','ind_S');
