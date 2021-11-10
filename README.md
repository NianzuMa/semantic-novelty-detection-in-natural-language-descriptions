# Semantic Novelty Detection in Natural Language Descriptions

* This repository is the original implementation of EMNLP 20201 Paper: [Semantic Novelty Detection in Natural Language Descriptions](https://aclanthology.org/2021.emnlp-main.66/)
* Please contact [@NianzuMa](https://github.com/NianzuMa) for questions and suggestions.
* For code running issues, please submit to the Github issues of this repository.
* The dataset can be downloaded this [drive](https://drive.google.com/drive/folders/133FkifkVGOhEu-slJy5YdBw7TIxvYH6O?usp=sharing).
* The dataset need to be parsed and processed. The owner of this repository is writing more details instructions to make sure the code can run smoothly.

![](./res/emnlp_2021.png)


## Citation
```
@inproceedings{ma-etal-2021-semantic,
    title = "Semantic Novelty Detection in Natural Language Descriptions",
    author = "Ma, Nianzu  and
      Politowicz, Alexander  and
      Mazumder, Sahisnu  and
      Chen, Jiahua  and
      Liu, Bing  and
      Robertson, Eric  and
      Grigsby, Scott",
    booktitle = "Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing",
    month = nov,
    year = "2021",
    address = "Online and Punta Cana, Dominican Republic",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2021.emnlp-main.66",
    pages = "866--882",
    abstract = "This paper proposes to study a fine-grained semantic novelty detection task, which can be illustrated with the following example. It is normal that a person walks a dog in the park, but if someone says {``}A man is walking a chicken in the park{''}, it is novel. Given a set of natural language descriptions of normal scenes, we want to identify descriptions of novel scenes. We are not aware of any existing work that solves the problem. Although existing novelty or anomaly detection algorithms are applicable, since they are usually topic-based, they perform poorly on our fine-grained semantic novelty detection task. This paper proposes an effective model (called GAT-MA) to solve the problem and also contributes a new dataset. Experimental evaluation shows that GAT-MA outperforms 11 baselines by large margins.",
}
```
