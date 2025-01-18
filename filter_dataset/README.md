# filter_dataset

This is a collection of scripts and notebooks to hand-label input-output pairs as useful or not, train a classifier on them, and then filter the dataset accordingly.

# Background

I found that Snapchat lenses typically only worked for about ~25% of the input images. The rest of the time, the lens did something trivial like change the saturation of the whole image, or slightly blur the background based on some primitive foreground detector.

I figured it was better to filter out all of the bad "no-op" pairs than to train on them. I found that a tiny convolutional neural network trained from scratch could achieve almost 100% accuracy (predicting if a pair is useful or not) with only a few hundred hand labels.
