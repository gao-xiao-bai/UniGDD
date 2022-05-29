# UniGDD: A Unified Generative Framework for Goal-Oriented Document-Grounded Dialogue


Code for paper [UniGDD: A Unified Generative Framework for Goal-Oriented Document-Grounded Dialogue](https://aclanthology.org/2022.acl-short.66/) (ACL 2022)

The goal-oriented document-grounded dialogue aims at responding to the user query based on the dialogue context and supporting document. Existing studies tackle this problem by decomposing it into two sub-tasks: knowledge identification and response generation. However, such pipeline methods would unavoidably suffer from the error propagation issue. This paper proposes to unify these two sub-tasks via sequentially generating the grounding knowledge and the response. We further develop a prompt-connected multi-task learning strategy to model the characteristics and connections of different tasks and introduce linear temperature scheduling to reduce the negative effect of irrelevant document information. Experimental results demonstrate the effectiveness of our framework.

## Prepare the environment

Please run the following commands:

``````
conda env create -f UniGDD.yaml
conda activate UniGDD
``````

## Prepare the data

Please run the following command:

``````
sh data.sh
``````

## Reproduce the results

Please run the following command:

``````
sh T5.sh
``````



## Citation
If you find the code helpful, please star this repo and cite our paper:
```
@inproceedings{gao-etal-2022-unigdd,
    title = "{U}ni{GDD}: {A} Unified Generative Framework for Goal-Oriented Document-Grounded Dialogue",
    author = "Gao, Chang  and
      Zhang, Wenxuan  and
      Lam, Wai",
    booktitle = "Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 2: Short Papers)",
    month = may,
    year = "2022",
    address = "Dublin, Ireland",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2022.acl-short.66",
    pages = "599--605",
    abstract = "The goal-oriented document-grounded dialogue aims at responding to the user query based on the dialogue context and supporting document. Existing studies tackle this problem by decomposing it into two sub-tasks: knowledge identification and response generation. However, such pipeline methods would unavoidably suffer from the error propagation issue. This paper proposes to unify these two sub-tasks via sequentially generating the grounding knowledge and the response. We further develop a prompt-connected multi-task learning strategy to model the characteristics and connections of different tasks and introduce linear temperature scheduling to reduce the negative effect of irrelevant document information. Experimental results demonstrate the effectiveness of our framework.",
}
```

