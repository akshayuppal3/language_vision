%[model_test model_crossval acc_crossval random_crossval]=embedding(label,embed,data,c,nfolds,nneg,seed)
%--------OVERVIEW-------
%Learns a bilinear alignment function with parameter A. Given two vectors x and y, x for "example" and y for "class", compute an alignment score x'Ay as how much x belongs to y.
%The setting for this function is that there are N examples belonging to K classes (example-class indicated by a NxK -1/1 label matrix). Each class is represented as a D1-dimensional embedding vector. Each example is represented as a D2-dimensional embedding vector. Randomly sample non-matching pairs, performs kernel expansion and learn A using liblinear, and perform cross validation for model selection.
%--------INPUTS-------
%label: the NxK -1/1 label matrix
%embed: KxD1 embedding vectors for each class (sparse)
%data: NxD2 features for examples (sparse)
%c: an array of C values for liblinear
%nfolds: number of folds for crossval
%nneg: number of negative samples
%seed: seed for negative sampling
%-------OUTPUTS-------
%model_test: a cell array of models good for testing
%model_crossval: an 2D cell array of crossval models
%acc_crossval: the performances of crossval models in AP
%random_crossval: the baseline performances in AP

function [model_test model_crossval acc_crossval random_crossval]=train_embedding(label,embed,data,c,nfolds,nneg,seed)
rng(seed);

acc_crossval=zeros(length(c),nfolds);
random_crossval=zeros(length(c),nfolds);


%negative sampling
pos=find(label(:)>0);
neg=randsample(find(label(:)<0),nneg);

%bilinear kernel expansion
feat_pos=cell(length(pos),1);
feat_neg=cell(length(neg),1);
for i=1:length(pos)
	id=pos(i);
	imid=mod(id-1,size(label,1))+1;
	wid=floor((id-1)/size(label,1))+1;
	feat_pos{i}=kron(embed(wid,:),data(imid,:));
end
for i=1:length(neg)
	id=neg(i);
	imid=mod(id-1,size(label,1))+1;
	wid=floor((id-1)/size(label,1))+1;
	feat_neg{i}=kron(embed(wid,:),data(imid,:));
end

label=label([pos(:);neg(:)]);
feat=[fast_sparse_vec_cell2mat(feat_pos);fast_sparse_vec_cell2mat(feat_neg)];

%crossval and training 
model_crossval=cell(length(c),nfolds);
model_test=cell(length(c),1);
for cid=1:length(c)
for foldid=1:nfolds
	fold_ind=foldid:nfolds:size(feat,1);
	fold_ind_inv=setdiff(1:size(feat,1),fold_ind);
	
	model_crossval{cid,foldid}=train(label(fold_ind_inv,1),sparse(feat(fold_ind_inv,:)),['-c ' num2str(c(cid))]);
	if length(model_crossval{cid,foldid}.Label)==2
		[~,~,scores_cval]=predict(label(fold_ind,1),sparse(feat(fold_ind,:)),model_crossval{cid,foldid});
		[acc_crossval(cid,foldid),random_crossval(cid,foldid)]=precision(scores_cval,label(fold_ind,1));
	else
		acc_crossval(cid,foldid)=NaN;
		random_crossval(cid,foldid)=0;
	end
end
model_test{cid}=train(label(:,1),sparse(feat(:,:)),['-c ' num2str(c(cid))]);
end

mean(acc_crossval)
mean(random_crossval)
clear feat_pos;
clear feat_neg;
clear feat;

function S=fast_sparse_vec_cell2mat(fv)

ind_array=cell(length(fv),3);
for i=1:length(fv)
	disp(num2str(i));
	ind_array{i,2}=find(fv{i})';
	ind_array{i,1}=i*ones(size(ind_array{i,2}));
	ind_array{i,3}=nonzeros(fv{i});
end

ind_1=cell2mat(ind_array(:,1));
ind_2=cell2mat(ind_array(:,2));
ind_3=cell2mat(ind_array(:,3));

S=sparse(ind_1,ind_2,ind_3,size(fv,1),length(fv{1}));

clear ind_1;
clear ind_2;
clear ind_3;