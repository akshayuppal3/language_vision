# Learning-Common-Sense-Through-Visual-Abstraction

Code and datasets for the commonsense assertion assessment methods from "Learning Common Sense Through Visual Abstraction". https://vision.ece.vt.edu/cs/

We ground commonsense assertions in abstract scenes and their tuple descirptions to assess the plausibility of the commonsense assertions.

## Dataset

**Commonsense assertions**

data/val.mat contains the (P)rimary, (R)elation and (S)econdary words of 14548 tuples, along with one human judgment score for each tuple. The score for each tuple is computed as the number of judges that think the tuple typically occurs minus the number of judges that think the tuple is not typical.

data/test.mat contains 14332 test tuples.

**Abstract scene tuple illustrations**

data/PRS_abstract_scenes.mat contains the tuples (P,R,S) and the image features of their corresponding abstract scene illustrations. There are 4260 tuples with abstract scene illustrations. The raw data of the abstract scene illustrations can be downloaded separately from https://vision.ece.vt.edu/cs/.

## Dependencies

This code is written in MATLAB. You'll need to install liblinear with its MATLAB wrapper in external/ to run this code. 

## Evaluating pretrained models

Run the following commands in MATLAB to evaluate the pretrained models.

    test=load('data/test.mat');
    w2v=load('data/coco_w2v.mat');
    abstract_scenes=load('data/PRS_abstract_scenes.mat');
    PRS_model=load('model/clipart_PRS.mat');
    cs_models=load('model/cs_models.mat');
    evaluate_cs_models(test,w2v,abstract_scenes,PRS_model,cs_models)

It should print the results reported in the paper.

    Visual Only
    AP: 0.68714
    Rank Corr: 0.45279
    Text Only
    AP: 0.7222
    Rank Corr: 0.49043
    Visual+Text
    AP: 0.73621
    Rank Corr: 0.50033

## Training models from scratch

Run the following commands in MATLAB to train the abstract scene grounding models and our joint text-vision plausibility assessment models from scratch.

    #Train abstract scene grounding models
    abstract_scenes=load('data/PRS_abstract_scenes.mat');
    w2v=load('data/coco_w2v.mat');
	learn_PRS_models(abstract_scenes,w2v)
	
    #Train commonsense assertion models
    val=load('data/val.mat');
    PRS_model=load('model/clipart_PRS.mat');
	learn_cs_models(val,w2v,abstract_scenes,PRS_model)
	
	#Evaluate
    test=load('data/test.mat');
    cs_models=load('model/cs_models.mat');
    evaluate_cs_models(test,w2v,abstract_scenes,PRS_model,cs_models)

Training the abstract scene grounding models usually takes about 2 hours. Training the commonsense assertion models takes about 5 minutes. They should end up the same as the provided pretrained models.

## Reference

If you find our datasets and code useful, please cite the following paper:

Ramakrishna Vedantam, Xiao Lin, Tanmay Batra, C. Lawrence Zitnick, Devi Parikh. **"Learning Common Sense Through Visual Abstraction"** In *ICCV*, 2015.

    @InProceedings{vedantamLinICCV15,
      author = {Ramakrishna Vedantam and Xiao Lin and Tanmay Batra and C. Lawrence Zitnick and Devi Parikh},
      title = {Learning Common Sense Through Visual Abstraction},
      booktitle = {International Conference on Computer Vision (ICCV)},
      year = {2015}
    }
