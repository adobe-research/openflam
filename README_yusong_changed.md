In API:

`get_local_similarity`: to perform the sound event detection using FLAM. The output will be a frame-wise similarity matrix. If `cross_product=True`, we will run similarity of each audio with every text class, otherwise each audio will only detected by the corresponding text. You can also select the similarity detection method using `method`, from `unbiased` (the true unbiased classifier, eq. 7 in paper) and `approximate` (eq. 8 that does not need to compute the per-logit bias).

Loss:

We provide the implementation of original InfoNCE loss, FLAM frame-wise contrastive loss and prior loss in OpenFLAM-main/src/openflam/module/contrastive_loss.py . We do not provide the code to train the model nor the dataloader. You can check the `loss` method of `FLAM` class `OpenFLAM-main/src/openflam/module/model.py` to get a sense of how we provide the data to the model.

SED inference and plot example:
`sed_inference_and_plot.py` provide an example to run open-vocab SED detection on an audio file using FLAM model, plot the results and save the plot as PNG to output directory.
