2024-02-06
- Get access to the DIKU cluster (done)
- Read TSGBench paper (https://arxiv.org/abs/2309.03755) (done)
- Get TSGBench (https://github.com/YihaoAng/TSGBench) to work with some of our own generated data (done)
- Create contained environment (apptainer) for one of the model (any) - apptainer recipe - can be tried locally first (done)
- Get the environment/containers runnning on the cluster (done)

2024-03-04
- Get TSGBench (https://github.com/YihaoAng/TSGBench) to work with some of our own generated data (done)
- Get the environment/containers runnning on the cluster (done)
- Get at least one model working on the cluster (done)
- Extract real data curves (done)
- Upload data to the cluster 
- Revisit GenLive (done)
- Create one-page abstract draft (ongoing)

2024-03-11
- actually make plots with metrics work based on pkl file of simulated and real data (done: for sine waves)
- models working - TimeGAN, TimeVAE - evaluate those two with simulated and real data with plots above (TimeVAE: done, TimeGAN: ongoing)
- models not working yet - RGAN, TSGM -> drop TSGM for now since there seem to be some misunderstandings wrt the TSGMBench article and what they reference
- fix the issue of per participants sample length (different lenghts of restfMRI scans), either per batch or make a global decision (done)
- look at participant data, make histogram of sequence length (done)
- choose another mixed-type model besides TSGM, look around (done: Fourier Flow)
- set up Github with code properly following instructions from Mel (done)
- Upload anonymized data to the cluster using randomly generated subject ids, time series, age and gender 

2024-03-18
- Generate correlated sine waves. (done: two different methods)
- Finalize the TimeGAN and TimeVAE pipeline. (done)
- Focus on report writting:
    - complete the Introduction section. (done)
    - extract relevant information from the POCS and put it into the Background and Methods sections.
        - includes generation of data, algorithms and data descriptions (sines and fmri) (done: sines)
    - write a sub-chapter on code reproducability. (done)
    - include plots of sine data (done) and fmri data
- Extras:
    - We have implemented a hyper-parameter search for RGAN, TimeGAN, TimeVAE, and Fourier-flows.
    - RGAN is running on the cluster (sines).
    - Fourier-flows is running on the cluster (sines).
 
2024-04-02
- According to the POCS feedback, remove unrealistic outliers from training data, use mean +-2 stds only. (done)
- Convert degrees/radians to mm by assuming a 10 cm diameter aka 50 mm radius sphere, see POCS. (done)
- Start hyper paramater search first on model specific parameters and then on classical dl paramters, evaluation criteria for good performance is TSGBench - area of spiderweb? (done)
- Address Melanie's comments to the thesis writing and add images. (thesis writting: done, images: ongoing)
- Melanie will add POCS to GitHub repo -> done: https://github.com/melanieganz/MoCoProject/tree/main/ChildrenHeadMotionABCD

Evaluation for the first stage of the grid search is based on visual examination of the t-SNE plots and the KL Divergence. We have modified TimeGAN's visualization to add such functionality to TimeVAE and Fourier-flows.

2024-04-08
- Melanie will find info for CFID.
- finish discussion of evaluation metrics in the report
- add two models to report TimeVAE (done), RGAN (done), TimeGAN (ongoing) FourirerFlow to report
- stop grid search for TimeGan in stage 1, continue with stage 2 for the others (training protocol was updated according to Vincent's suggestions)
- regenerate t-sne plots, 10 times 1000 points randomly sampeld with replacement
- add figures genereated last week to report
- add stage figures to report as well

2024-04-22
- Sine experiments are ran locally for RGAN, TimeGAN, TimeVAE, and Fourier Flow on 10,000 samples of:
    - single sine wave with 24 sampling points;
    - six correlated sine waves with 24 sampling points;
- TimeVAE and Fourier Flow are evaluated on 10,000 samples of six correlated sine waves with 380 sampling points.
- RGAN was evaluated on 10,000 samples of a smooth function with 72 sampling points (TimeGAN: ongoing).
- Grid search and first stage of evaluation for Fourier Flow are done, currently training the final 9 models (7/9).
- Extended the Methods & Results section to reflect our sine experiments, the Discussion section includes some of the talking points.

2024-05-06
- Merge Jakob's and Hristo's discussion points. (done)
- Produce and add mean power spectrum for RGAN and TimeGAN.
- Finish captioning the figures and tables (by 10th of May). (done)
- Finish the fMRI Results section (by 12th of May). (done)
- Start the fMRI Discussion and (potentially) finish it (by 12th of May).

