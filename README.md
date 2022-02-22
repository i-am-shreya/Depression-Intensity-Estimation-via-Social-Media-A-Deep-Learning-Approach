# Depression-Intensity-Estimation-via-Social-Media-A-Deep-Learning-Approach

Depression has become a big problem in our society today. It is also a major reason for suicide, especially among teenagers. In the current outbreak of coronavirus disease (COVID-19), the affected countries have recommended social distancing and lockdown measures. Resulting in interpersonal isolation, these measures have raised serious concerns for mental health and depression. Generally, clinical psychologists diagnose depressed people via face-to-face interviews following the clinical depression criteria. However, often patients tend to not consult doctors in their early stages of depression. Nowadays, people are increasingly using social media to express their moods. In this article, we aim to predict depressed users as well as estimate their depression intensity via leveraging social media (Twitter) data, in order to aid in raising an alarm. We model this problem as a  self-supervised learning task. We start with weakly labeling the Twitter data in a self-supervised manner. A rich set of features, including emotional, topical, behavioral, user level, and depression-related n-gram features, are extracted to represent each user. Using these features, we train a small long short-term memory (LSTM) network using Swish as an activation function, to predict the depression intensities.

!(/figs/depression_intensity.png) Depression intensity analysis from social media on BDI-II scale.

If you find the paper/code useful for your research, please consider citing our work:
```
@article{ghosh2021depression,
  title={Depression Intensity Estimation via Social Media: A Deep Learning Approach},
  author={Ghosh, Shreya and Anwar, Tarique},
  journal={IEEE Transactions on Computational Social Systems},
  volume={8},
  number={6},
  pages={1465--1474},
  year={2021},
  publisher={IEEE}
}
```
## Contact
- <a href="https://sites.google.com/view/shreyaghosh/home">Shreya Ghosh</a> and <a href="https://scholar.google.com/citations?user=pomKKtoAAAAJ&hl=en">Dr Tarique Anwar</a>.

## Acknowledgements
This repository makes liberal use of data repositarites from 
[MDDL](https://github.com/sunlightsgy/MDDL), [NLTK](https://www.nltk.org/index.html).
