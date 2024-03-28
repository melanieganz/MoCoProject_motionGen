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
    - We have implemented a hyper-parameter search for TimeVAE and TimeGAN.
    - TimeGAN can now generate a distribution the size of the testing set (it was previously fixed to the training set's size).
    - RGAN is running locally, but it has GPU/CUDA-related issues on the cluster.
    - Fourier-flows. (trains properly, but cannot sample during evaluation)
