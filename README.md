# Generative Motion Models for Rigid Head Motion of Children

Head motion is one of the major reasons for artefacts in Magnetic Resonance
Imaging (MRI), which is especially challenging for children who are often intimidated
by the dimensions of the MR scanner. To optimize the MRI acquisition
for children in the clinical setting, insights into children’s motion patterns are
essential. The ABCD study is the largest long-term study of brain development
and child health in the United States. The National Institutes of Health (NIH)
funded leading research in adolescent development and neuroscience to conduct
this ambitious project.

The ABCD Research Consortium comprises a Coordinating Center, a Data
Analysis, Informatics and Resource Center, and 21 research sites across the
United States, inviting 11,880 children ages 9-10 to join the study. Understanding
the prevalence and degree of head motion in children can help guide
the development of motion correction techniques for MRI. The project aims
to analyse the motion logs from the diffusion imaging and resting-state fMRI
series of the ABCD study to understand how far a typical child moves during
scanning, analyse correlations between motion parameters, and assess the rank
of the combined motion matrices. The data utilised will be resting-state fMRI
(2-5 min. runs with 10 sec. film clip between runs) and diffusion tensor imaging
(9-10 minutes with a movie shown).

Based on the analysis of the motion curves of the ABCD study, the project
aims to develop an algorithm using a Generative Adversarial Network (GAN)
to obtain transformation matrices with the same statistics as the ABCD’s motion
logs. This can be used in the future to optimize the MRI acquisition for
children. The generative model will be evaluated on both a subset of the ABCD
dataset and an in-house dataset of children’s MRI scans acquired by the Motion
Correction (MoCo) project at Rigshospitalet, Denmark, where high-temporal
resolution motion tracking is available.
