# 3 ViT, 1 Decoder no data augmentation
- SSL 0.259
- bug with positional embeddings BEFORE projection into D

# 3 ViT, 1 Decoder mae_pretrain_aug_0.208_192, 200 epochs
- SSL 0.208 training loss
- same as above but with data agumentation (random cropping and flip)

# 6 ViT Encoder Blocks, 1 Decoder, D=384, Batch=256, 200 epochs
- SSL training loss of 0.19122
- linear probe accuracy of 0.6326

# 8 ViT Encoder Blocks, 2 Decoder, D=384, Batch=256, 200 epochs
- SSL training loss of 0.1848
- linear probe accuracy of 0.6787
- D=384, D=384 AND 3 heads

# ViT-B/8, 8 Decoder, 400 epochs, Autocast, Gradient Checkpointing
- SSL training loss of 0.1782, 0.177 (see latest checkpoint at 160)

# ViT-S/8, 12e, 4d, 400 epochs, no autocast, gradient checkpointing

NEXT:
- X Fine-tune the linear probe on 8e_2d & visualize & Record!
- X try torch.autocast to float16 + grad checkpointing and train ViT-Base
- X train ViT-Base and see how well we do
- X train ViT-Small (100k may be too small for ViT-Base) & bump BatchSize

THIS IS SUPER COOL! FOR LINEAR PROBING
- precompute encoder's output for ALL DATA POINTS THEN FIT cyanure (convex optimization)-
-> No learning rate, just pops out the optimal convex solution fitted via logistic regression!

- linear probe VIT-Small (w/ BatchNorm no affine)
- train ResNet tiny fully supervised on STL10 (no unsupervised dataset) & compare?
- train a Vit-Tiny fully supervised on STL10 & compare?
- try fine-tuning the entire networks ViT-Small 
- train a ViT-Tiny model to compare

- Create a short report on my ablation studies
- Create a list of tricks section of the report for myself
- Dive into the hyperparameters: AdamW, linear probe training too

INTERESTING:
- They apply random resize crops+horizontal flips augmentation during linear probe evaluations!
- 

The learning rate I use is right out of the DINO paper
- AdamW
- 10 epochs to linearly ramp up the LR
- Cosine schedule after that
- weight decay 0.04 to 0.4


THINGS TO TRY
1. Try with appropriate nubmer of heads for decoder
2. Try decoder dim=256, 4 heads, depth 4
3. Try with correct embeddings normalization

Ok possibly something wrong is the decoder had too many heads which may have screwed up learning
I could also try 

- smaller batch size SO that I don't need gradient checkpoint and it trains faster