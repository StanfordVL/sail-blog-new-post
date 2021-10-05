---
layout: post
title: "Building Scalable, Explainable, and Adaptive NLP Models with Retrieval"
short-summary: "By tapping into knowledge stored explicitly in text corpora, retrieval helps tackle the inefficiency, opaqueness, and static nature of large language models."
summary: "By tapping into knowledge stored explicitly in text corpora, retrieval helps tackle the inefficiency, opaqueness, and static nature of large language models."
feature-img: "/assets/img/posts/2021-10-01-retrieval-based-NLP/image1.png"
thumbnail: "/assets/img/posts/2021-10-01-retrieval-based-NLP/late-interaction.png"
author: <a href="https://omarkhattab.com/">Omar Khattab</a>, <a href="http://web.stanford.edu/~cgpotts/">Christopher Potts</a>, and <a href="https://cs.stanford.edu/~matei/">Matei Zaharia</a>
tags: [nlp, ir, natural language processing, information retrieval, open-domain question answering, multi-hop reasoning, colbert, colbert-qa, baleen]
draft: True
---


Natural language processing (NLP) has witnessed impressive developments
in answering questions, summarizing or translating reports, and
analyzing sentiment or offensiveness. Much of this progress is owed to
training ever-larger language models, such
as **[T5](https://arxiv.org/abs/1910.10683)** or **[GPT-3](https://arxiv.org/abs/2005.14165)**,
that use deep monolithic architectures to internalize how language is
used within text from massive Web crawls. During training, these models
distill the facts they read into _implicit knowledge_, storing in their
parameters not only the capacity to “understand” language tasks, but
also highly abstract _knowledge representations_ of entities, events, and
facts the model needs for solving tasks.

Despite the well-publicized success of large language models, their
black-box nature hinders key goals of NLP. In particular, existing large
language models are generally:

-   **Inefficient.** Researchers continue to enlarge these models, leading
    to striking inefficiencies as the field already pushes past 1
    trillion parameters. This imposes a considerable [environmental impact](https://arxiv.org/abs/1906.02243)
    and its costs exclude all but a few large organizations from the
    ability to train—or in many cases even deploy—such models.

-   **Opaque.** They encode “knowledge” into model weights, synthesizing
    what they manage to memorize from training examples. This makes it
    difficult to discern what sources—if any—the model uses to make a
    prediction, a concerning problem in practice as these models
    frequently generate fluent yet untrue [statements](https://twitter.com/yoavgo/status/1284192413477670912).

-   **Static.** They are expensive to update. We cannot efficiently adapt a
    GPT model trained on, say, Wikipedia text from 2019 so it reflects
    the knowledge encoded in the 2021 Wikipedia—or the latest snapshot
    of the medical preprint server medRXiv. In practice, adaptation often
    necessitates expensive retraining or [fine-tuning](https://arxiv.org/abs/2106.15110) on the new corpus.

This post explores an emerging alternative, **Retrieval-based NLP**, in
which models directly “search” for information in a text corpus to
exhibit knowledge, leveraging the representational strengths of language models
while addressing the challenges above. Such
models—including **[REALM](https://arxiv.org/abs/2002.08909)**, **[RAG](https://arxiv.org/abs/2005.11401)**, **[ColBERT-QA](https://arxiv.org/abs/2007.00814)**,
and **[Baleen](https://arxiv.org/abs/2101.00436)**—are
already advancing the state of the art for tasks like answering
open-domain questions and verifying complex claims, all with
architectures that back their predictions with checkable sources while
being 100–1000× smaller, and thus far cheaper to execute, than GPT-3. At
Stanford, we have shown that improving the expressivity and
supervision of scalable neural retrievers can lead to much stronger NLP
systems: for instance, **ColBERT-QA** improves answer correctness on open-QA
benchmarks by up to **16** EM points and **Baleen** improves the ability to
check complex claims on
**[HoVer](https://hover-nlp.github.io/)**,
correctly and with provenance, by up to **42** percentage points against existing work.



### Retrieval-based NLP


{% figure %}
<img class="postimage_75" src="{{ site.baseurl }}/assets/img/posts/2021-10-01-retrieval-based-NLP/image1.png"/>
<figcaption>
Figure 1: An illustration comparing (a) black-box language models and (b) retrieval-oriented NLP models, the paradigm this post advocates for.
</figcaption>
{% endfigure %}


As **Figure 1** illustrates, retrieval-based NLP methods view tasks as
“[open-book](https://arxiv.org/abs/1809.02789)”
exams: knowledge is encoded _explicitly_ in the form of a text corpus like
Wikipedia, the medical literature, or a software’s API documentation. When
solving a language task, the model **learns to search** for pertinent passages
and to then **use the retrieved information** for crafting knowledgeable responses.
In doing so, retrieval helps decouple the capacity that language models have for
_understanding text_ from how they _store knowledge_, leading to three key advantages.

<!-- ### Benefits of retrieval-based NLP models -->

**Tackling Inefficiency.** Retrieval-based models can be **much smaller and
faster**, and thus more environmentally friendly. Unlike black-box language models,
the parameters no longer need to store an ever-growing list of facts, as
such facts can be retrieved. Instead, we can dedicate those parameters
for processing language and solving tasks, leaving us with smaller
models that are highly effective. For instance, ColBERT-QA achieves
**47.8%** EM on the open-domain Natural Questions task, whereas a fine-tuned
T5-11B model (with **24x** more parameters) and a few-shot GPT-3 model (with
**400x** more parameters) achieve only **34.8%** and **29.9%**, respectively.

**Tackling Opaqueness.** Retrieval-based NLP offers a **transparent** contract
with users: when the model produces an answer, we can read the sources
it retrieved and judge their relevance and credibility for ourselves.
This is essential whether the model is factually correct or not: by
inspecting the sources surfaced by a system like Baleen, we can trust
its outputs only if we find that reliable sources do support them.

**Tackling Static Knowledge.** Retrieval-based models emphasize learning
general techniques for finding and connecting information from the
available resources. With facts stored as text, the retrieval knowledge
store can be **efficiently updated or expanded** by modifying the text
corpus, all while the model’s capacity for finding and using information
remains constant. Besides computational cost reductions, this expedites generality:
developers, even in niche domains, can “plug in” a domain-specific text
collection and rely on retrieval to facilitate domain-aware responses.



### ColBERT: Scalable yet expressive neural retrieval

As the name suggests, retrieval-based NLP relies on semantically rich **search** to extract
information. For search be practical and effective, it must scale to massive text corpora.
To draw on the open-book exam analogy, it’s hopeless to linearly look
through the pages of a hefty textbook during the exam—we need scalable
strategies for organizing the content in advance, and efficient
techniques for locating relevant information at inference time.


{% figure %}
<img class="postimage_75" src="{{ site.baseurl }}/assets/img/posts/2021-10-01-retrieval-based-NLP/image2.png"/>
<figcaption>
Figure 2: Schematic diagrams comparing two popular paradigms in neural IR in sub-figures (a) and (b) against the late interaction paradigm of ColBERT in sub-figure (c).
</figcaption>
{% endfigure %}



Traditionally in IR, search tasks were conducted using bag-of-words
models like BM25, which seek documents that contain the same tokens as
the query. In
2019, [search](https://arxiv.org/abs/1901.04085) [was](https://arxiv.org/abs/1910.10687) [revolutionized](https://arxiv.org/abs/1904.07094) with **[BERT](https://arxiv.org/abs/1810.04805)** for
ranking and its deployment
in [Google](https://blog.google/products/search/search-language-understanding-bert/) and [Bing](https://azure.microsoft.com/en-us/blog/bing-delivers-its-largest-improvementin-search-experience-using-azure-gpus/) for
Web search. The standard approach is illustrated in **Figure 2(a)**. Each
document is concatenated with the query, and both are fed jointly into a BERT
model, fine-tuned to estimate relevance. BERT _doubled_ the MRR@10 quality
metric over BM25 on the popular MS MARCO Passage Ranking leaderboard,
but it simultaneously posed a fundamental limitation: scoring
_each_ query–document pair requires _billions_ of computational operations
(FLOPs). As a result, BERT can only be used to _re-rank_ the top-k (e.g.,
top-1000) documents already extracted by simpler methods like BM25,
having no capacity to recover useful documents that bag-of-word search
misses.

The key limitation of this approach is that it encodes queries and
documents _jointly_. Many **representation-similarity** systems have been
proposed to tackle this, some of which re-purpose BERT within the
paradigm depicted in **Figure 2(b)**. In these systems
(like **[SBERT](https://arxiv.org/abs/1908.10084)** and **[ORQA](https://arxiv.org/abs/1906.00300)**,
and more
recently **[DPR](https://arxiv.org/abs/2004.04906)** and **[ANCE](https://arxiv.org/abs/2007.00808)**,
every document in the corpus is fed into a BERT encoder that produces a
dense vector meant to capture the semantics of the document. At search
time, the query is encoded, separately, through another BERT encoder, and the
top-k related documents are found using a dot product between the query
and document vectors. By removing the expensive interactions between the
query and the document, these models are able to scale far more
efficiently than the approach in **Figure 2(a)**.

Nonetheless, representation-similarity models suffer from an
architectural bottleneck: they encode the query and document into
coarse-grained representations and model relevance as a single dot
product. This greatly diminishes quality compared with expensive
re-rankers that model token-level interactions between the contents of
queries and documents. Can we efficiently scale fine-grained, contextual
interactions to a massive corpus, without compromising speed or quality?
It turns out that the answer is “yes”, using a paradigm called late
interaction, first devised in
our **[ColBERT](https://arxiv.org/abs/2004.12832)**[^colbert] [[code](https://github.com/stanford-futuredata/ColBERT)]
model, which appeared at SIGIR 2020.

As depicted in **Figure 2(c)**, **ColBERT** independently encodes queries and
documents into fine-grained **multi-vector representations**. It then
attempts to softly and contextually locate each query token inside the
document: for each query embedding, it finds the most similar embedding
in the document with a “MaxSim” operator and then sums up all of the
MaxSims to score the document. “MaxSim” is a careful choice that allows
us to index the document embeddings for [Approximate Nearest Neighbor](https://arxiv.org/abs/1702.08734)
(ANN) search, enabling us to scale this rich interaction to millions of passages with latency
on the order of tens of milliseconds. For instance, ColBERT can search over all
passages in English Wikipedia in approximately **70 milliseconds** per query.
On MS MARCO Passage Ranking, ColBERT preserved the MRR@10 quality of BERT re-rankers while boosting recall@1k to nearly **97%**
against the official BM25 ranking's recall@1k of just **81%**.

Making neural retrievers more lightweight remains an active area of
development, with models like **[DeepImpact](https://arxiv.org/abs/2104.12016)**
that trade away some quality for extreme forms of efficiency and
developments like **[BPR](https://arxiv.org/abs/2106.00882)**
and **[quantized ColBERT](https://github.com/stanford-futuredata/ColBERT/tree/binarization)**
that reduce the storage footprint by an order of magnitude while
preserving the quality of DPR and ColBERT, respectively.



### ColBERT-QA and Baleen: Specializing neural retrieval to complex tasks, with tracked provenance

While scaling expressive search mechanisms is critical, NLP models need
more than just finding the right documents. In particular, we want NLP models
to use retrieval to answer questions, fact-check claims, respond
informatively in a conversation, or identify the sentiment of a piece of
text. Many tasks of this kind—dubbed _knowledge-intensive_ language
tasks—are collected in
the **[KILT](https://ai.facebook.com/tools/kilt/)** benchmark.
The most popular task is open-domain question answering (or Open-QA).
Systems are given a question from any domain and must produce an answer,
often by reference to the passages in a large corpus, as depicted in
**Figure 1(b)**.




{% figure %}

<table style="border-collapse:collapse;border-color:#ccc;border-spacing:0;" class="tg"><colgroup><col style=""><col style=""><col style=""><col style=""><col style=""></colgroup><thead><tr><th style="background-color:#f0f0f0;border-color:inherit;border-style:solid;border-width:1px;color:#333;font-family:Arial, sans-serif;font-size:14px;font-weight:bold;overflow:hidden;padding:10px 5px;text-align:center;vertical-align:middle;word-break:normal">Benchmark</th><th style="background-color:#f0f0f0;border-color:inherit;border-style:solid;border-width:1px;color:#333;font-family:Arial, sans-serif;font-size:14px;font-weight:bold;overflow:hidden;padding:10px 5px;text-align:center;vertical-align:middle;word-break:normal">System</th><th style="background-color:#f0f0f0;border-color:inherit;border-style:solid;border-width:1px;color:#333;font-family:Arial, sans-serif;font-size:14px;font-weight:bold;overflow:hidden;padding:10px 5px;text-align:center;vertical-align:top;word-break:normal">Metric</th><th style="background-color:#f0f0f0;border-color:inherit;border-style:solid;border-width:1px;color:#333;font-family:Arial, sans-serif;font-size:14px;font-weight:bold;overflow:hidden;padding:10px 5px;text-align:center;vertical-align:top;word-break:normal">Gains</th><th style="background-color:#f0f0f0;border-color:inherit;border-style:solid;border-width:1px;color:#333;font-family:Arial, sans-serif;font-size:14px;font-weight:bold;overflow:hidden;padding:10px 5px;text-align:center;vertical-align:middle;word-break:normal">Baselines</th></tr></thead><tbody><tr><td style="background-color:#fff;border-color:inherit;border-style:solid;border-width:1px;color:#333;font-family:Arial, sans-serif;font-size:14px;font-weight:bold;overflow:hidden;padding:10px 5px;text-align:center;vertical-align:middle;word-break:normal" colspan="5">Open-Domain Question Answering</td></tr><tr><td style="background-color:#fff;border-color:inherit;border-style:solid;border-width:1px;color:#333;font-family:Arial, sans-serif;font-size:14px;font-weight:bold;overflow:hidden;padding:10px 5px;text-align:left;vertical-align:middle;word-break:normal">Open-NaturalQuestions</td><td style="background-color:#f9f9f9;border-color:inherit;border-style:solid;border-width:1px;color:#333;font-family:Arial, sans-serif;font-size:14px;overflow:hidden;padding:10px 5px;text-align:left;vertical-align:middle;word-break:normal" rowspan="3">ColBERT-QA</td><td style="background-color:#fff;border-color:inherit;border-style:solid;border-width:1px;color:#333;font-family:Arial, sans-serif;font-size:14px;overflow:hidden;padding:10px 5px;text-align:left;vertical-align:middle;word-break:normal" rowspan="3">Answer Match</td><td style="background-color:#f9f9f9;border-color:inherit;border-style:solid;border-width:1px;color:#333;font-family:Arial, sans-serif;font-size:14px;overflow:hidden;padding:10px 5px;text-align:left;vertical-align:top;word-break:normal">+3</td><td style="background-color:#fff;border-color:inherit;border-style:solid;border-width:1px;color:#333;font-family:Arial, sans-serif;font-size:14px;overflow:hidden;padding:10px 5px;text-align:left;vertical-align:middle;word-break:normal" rowspan="3">RAG, DPR, REALM, BM25+BERT</td></tr><tr><td style="background-color:#fff;border-color:inherit;border-style:solid;border-width:1px;color:#333;font-family:Arial, sans-serif;font-size:14px;font-weight:bold;overflow:hidden;padding:10px 5px;text-align:left;vertical-align:middle;word-break:normal">Open-TriviaQA</td><td style="background-color:#f9f9f9;border-color:inherit;border-style:solid;border-width:1px;color:#333;font-family:Arial, sans-serif;font-size:14px;overflow:hidden;padding:10px 5px;text-align:left;vertical-align:top;word-break:normal">+12</td></tr><tr><td style="background-color:#fff;border-color:inherit;border-style:solid;border-width:1px;color:#333;font-family:Arial, sans-serif;font-size:14px;font-weight:bold;overflow:hidden;padding:10px 5px;text-align:left;vertical-align:middle;word-break:normal">Open-SQuAD</td><td style="background-color:#f9f9f9;border-color:inherit;border-style:solid;border-width:1px;color:#333;font-family:Arial, sans-serif;font-size:14px;overflow:hidden;padding:10px 5px;text-align:left;vertical-align:top;word-break:normal">+17</td></tr><tr><td style="background-color:#fff;border-color:inherit;border-style:solid;border-width:1px;color:#333;font-family:Arial, sans-serif;font-size:14px;font-weight:bold;overflow:hidden;padding:10px 5px;text-align:center;vertical-align:middle;word-break:normal" colspan="5">Multi-Hop Reasoning</td></tr><tr><td style="background-color:#fff;border-color:inherit;border-style:solid;border-width:1px;color:#333;font-family:Arial, sans-serif;font-size:14px;font-weight:bold;overflow:hidden;padding:10px 5px;text-align:left;vertical-align:middle;word-break:normal" rowspan="2">HotPotQA</td><td style="background-color:#f9f9f9;border-color:inherit;border-style:solid;border-width:1px;color:#333;font-family:Arial, sans-serif;font-size:14px;overflow:hidden;padding:10px 5px;text-align:left;vertical-align:middle;word-break:normal" rowspan="4">Baleen</td><td style="background-color:#fff;border-color:inherit;border-style:solid;border-width:1px;color:#333;font-family:Arial, sans-serif;font-size:14px;overflow:hidden;padding:10px 5px;text-align:left;vertical-align:top;word-break:normal">Retrieval Success@20</td><td style="background-color:#f9f9f9;border-color:inherit;border-style:solid;border-width:1px;color:#333;font-family:Arial, sans-serif;font-size:14px;overflow:hidden;padding:10px 5px;text-align:left;vertical-align:top;word-break:normal">+10 / NA</td><td style="background-color:#fff;border-color:inherit;border-style:solid;border-width:1px;color:#333;font-family:Arial, sans-serif;font-size:14px;overflow:hidden;padding:10px 5px;text-align:left;vertical-align:middle;word-break:normal" rowspan="2">MDR / IRRR</td></tr><tr><td style="background-color:#fff;border-color:inherit;border-style:solid;border-width:1px;color:#333;font-family:Arial, sans-serif;font-size:14px;overflow:hidden;padding:10px 5px;text-align:left;vertical-align:top;word-break:normal">Passage-Pair Match</td><td style="background-color:#f9f9f9;border-color:inherit;border-style:solid;border-width:1px;color:#333;font-family:Arial, sans-serif;font-size:14px;overflow:hidden;padding:10px 5px;text-align:left;vertical-align:top;word-break:normal">+5 / +3</td></tr><tr><td style="background-color:#fff;border-color:inherit;border-style:solid;border-width:1px;color:#333;font-family:Arial, sans-serif;font-size:14px;font-weight:bold;overflow:hidden;padding:10px 5px;text-align:left;vertical-align:middle;word-break:normal" rowspan="2">HoVer</td><td style="background-color:#fff;border-color:inherit;border-style:solid;border-width:1px;color:#333;font-family:Arial, sans-serif;font-size:14px;overflow:hidden;padding:10px 5px;text-align:left;vertical-align:top;word-break:normal">Retrieval Success@100</td><td style="background-color:#f9f9f9;border-color:inherit;border-style:solid;border-width:1px;color:#333;font-family:Arial, sans-serif;font-size:14px;overflow:hidden;padding:10px 5px;text-align:left;vertical-align:top;word-break:normal">+48 / +17</td><td style="background-color:#fff;border-color:inherit;border-style:solid;border-width:1px;color:#333;font-family:Arial, sans-serif;font-size:14px;overflow:hidden;padding:10px 5px;text-align:left;vertical-align:middle;word-break:normal">TF-IDF / ColBERT-Hop</td></tr><tr><td style="background-color:#fff;border-color:inherit;border-style:solid;border-width:1px;color:#333;font-family:Arial, sans-serif;font-size:14px;overflow:hidden;padding:10px 5px;text-align:left;vertical-align:top;word-break:normal">“HoVer Score” for<br>Claim Verification<br>with Provenance</td><td style="background-color:#f9f9f9;border-color:inherit;border-style:solid;border-width:1px;color:#333;font-family:Arial, sans-serif;font-size:14px;overflow:hidden;padding:10px 5px;text-align:left;vertical-align:top;word-break:normal">+42</td><td style="background-color:#fff;border-color:inherit;border-style:solid;border-width:1px;color:#333;font-family:Arial, sans-serif;font-size:14px;overflow:hidden;padding:10px 5px;text-align:left;vertical-align:middle;word-break:normal">Official “TF-IDF + BERT” Baseline</td></tr><tr><td style="background-color:#fff;border-color:inherit;border-style:solid;border-width:1px;color:#333;font-family:Arial, sans-serif;font-size:14px;font-weight:bold;overflow:hidden;padding:10px 5px;text-align:center;vertical-align:middle;word-break:normal" colspan="5" rowspan="2">Cross-Lingual Open-Domain Question Answering    </td></tr><tr></tr><tr><td style="background-color:#fff;border-color:inherit;border-style:solid;border-width:1px;color:#333;font-family:Arial, sans-serif;font-size:14px;font-weight:bold;overflow:hidden;padding:10px 5px;text-align:left;vertical-align:middle;word-break:normal">XOR TyDi</td><td style="background-color:#f9f9f9;border-color:inherit;border-style:solid;border-width:1px;color:#333;font-family:Arial, sans-serif;font-size:14px;overflow:hidden;padding:10px 5px;text-align:left;vertical-align:middle;word-break:normal">GAAMA with ColBERT<br>from IBM Research</td><td style="background-color:#fff;border-color:inherit;border-style:solid;border-width:1px;color:#333;font-family:Arial, sans-serif;font-size:14px;overflow:hidden;padding:10px 5px;text-align:left;vertical-align:top;word-break:normal">Recall@5000-tokens</td><td style="background-color:#f9f9f9;border-color:inherit;border-style:solid;border-width:1px;color:#333;font-family:Arial, sans-serif;font-size:14px;overflow:hidden;padding:10px 5px;text-align:left;vertical-align:top;word-break:normal">+10</td><td style="background-color:#fff;border-color:inherit;border-style:solid;border-width:1px;color:#333;font-family:Arial, sans-serif;font-size:14px;overflow:hidden;padding:10px 5px;text-align:left;vertical-align:middle;word-break:normal">Official “DPR + Vanilla Transformer” Baseline</td></tr><tr><td style="background-color:#fff;border-color:inherit;border-style:solid;border-width:1px;color:#333;font-family:Arial, sans-serif;font-size:14px;font-weight:bold;overflow:hidden;padding:10px 5px;text-align:center;vertical-align:middle;word-break:normal" colspan="5">Zero-Shot Information Retrieval</td></tr><tr><td style="background-color:#fff;border-color:inherit;border-style:solid;border-width:1px;color:#333;font-family:Arial, sans-serif;font-size:14px;font-weight:bold;overflow:hidden;padding:10px 5px;text-align:left;vertical-align:middle;word-break:normal">BEIR</td><td style="background-color:#f9f9f9;border-color:inherit;border-style:solid;border-width:1px;color:#333;font-family:Arial, sans-serif;font-size:14px;overflow:hidden;padding:10px 5px;text-align:left;vertical-align:middle;word-break:normal">ColBERT</td><td style="background-color:#fff;border-color:inherit;border-style:solid;border-width:1px;color:#333;font-family:Arial, sans-serif;font-size:14px;overflow:hidden;padding:10px 5px;text-align:left;vertical-align:top;word-break:normal">Recall@100</td><td style="background-color:#f9f9f9;border-color:inherit;border-style:solid;border-width:1px;color:#333;font-family:Arial, sans-serif;font-size:14px;overflow:hidden;padding:10px 5px;text-align:left;vertical-align:top;word-break:normal">Outperforms other off-the-shelf<br>dense retrievers on 13/17 tasks</td><td style="background-color:#fff;border-color:inherit;border-style:solid;border-width:1px;color:#333;font-family:Arial, sans-serif;font-size:14px;overflow:hidden;padding:10px 5px;text-align:left;vertical-align:middle;word-break:normal">DPR, ANCE, SBERT, USE-QA</td></tr></tbody></table>

<figcaption>
Table 1: Results of models using ColBERT, ColBERT-QA, and Baleen across a wide range of language tasks.
</figcaption>


{% endfigure %}



Two popular models in this space are **REALM** and **RAG**, which rely on the
ORQA and DPR retrievers discussed earlier. REALM and RAG jointly tune a
retriever as well as a reader, a modeling component that consumes the
retrieved documents and produces answers or responses. Take RAG as an
example: its reader is a generative BART model, which attends to the
passages while generating the target outputs. While they constitute
important steps toward retrieval-based NLP, REALM and RAG suffer from
two major limitations. First, they use the restrictive paradigm of
**Figure 2(b)** for retrieval, thereby sacrificing recall: they are often
unable to find relevant passages for conducting their tasks. Second,
when training the retriever, REALM and RAG collect documents by
searching for them inside the training loop and, to make this practical, they
freeze the document encoder when fine-tuning, restricting the model’s adaptation to the task.

**[ColBERT-QA](https://arxiv.org/abs/2007.00814)**[^colbert-qa] is an Open-QA system (published at TACL'21) that we built on
top of ColBERT to tackle both problems. By adapting ColBERT's expressive search to the task,
ColBERT-QA finds useful passages for a larger fraction of the questions and thus
enables the reader component to answer more questions correctly and with provenance.
In addition, ColBERT-QA introduces **relevance-guided supervision** (RGS),
a training strategy whose goal is to adapt a
retriever like ColBERT to the specifics of an NLP task like Open-QA. RGS
proceeds in discrete rounds, using the retriever trained in the previous
round to collect “positive” passages that are likely useful for the
reader—specifically, passages ranked highly by the latest version of the
retriever and that also overlap with the gold answer of the question—and
challenging “negative” passages. By converging to a high coverage of
positive passages and by effectively sampling hard negatives, ColBERT-QA
improves retrieval Success@20 by more than **5**-, **5**-, and **12**-point gains on
the open-domain QA settings of NaturalQuestions, TriviaQA, and SQuAD, and thus greatly
improves downstream answer match.

A more sophisticated version of the Open-QA task is **multi-hop reasoning**,
where systems must answer questions or verify claims by gathering
information from multiple sources. Systems in this space,
like **[GoldEn](https://arxiv.org/abs/1910.07000)**, **[MDR](https://arxiv.org/abs/2009.12756)**,
and **[IRRR](https://arxiv.org/abs/2010.12527)**,
find relevant documents and “hop” between them—often by running
additional searches—to find all pertinent sources. While these models
have demonstrated strong performance for two-hop tasks, scaling robustly
to more hops is challenging as the search space grows exponentially.

To tackle this, our **[Baleen](https://arxiv.org/abs/2101.00436)**[^baleen] system
(accepted as a Spotlight paper at NeurIPS'21) introduces a richer pipeline for
multi-hop retrieval: after each retrieval “hop”, Baleen summarizes the
pertinent information from the passages into a short context that is used
to inform future hops. In doing so, Baleen controls the search space
architecturally—obviating the need to explore each potential passage
at every hop—without sacrificing recall. Baleen also extends ColBERT’s
late interaction: it allows the representations of different documents
to “focus” on distinct parts of the same query, as each of those documents
in the corpus might satisfy a distinct aspect of the same complex query.
As a result of its more deliberate architecture and its stronger
retrieval modeling, Baleen saturates retrieval on the popular two-hop
HotPotQA benchmark (raising answer-recall@20 from **89%** by MDR to **96%**) and
dramatically improves performance on the harder four-hop claim
verification
benchmark [HoVer](https://hover-nlp.github.io/),
finding all required passages in **92%** of the examples—up from just **45%**
for the official baseline and **75%** for a many-hop flavor of ColBERT.

In these tasks, when our retrieval-based models make predictions, we can
inspect their underlying sources and decide whether we can trust the
answer. And when model errors stem from specific sources, those can be
removed or edited, and making sure models are faithful to such edits
is an [active area](https://arxiv.org/abs/2109.05052) of work.


### Generalizing models to new domains with robust neural retrieval

In addition to helping with efficiency and transparency, retrieval
approaches promise to make domain generalization and knowledge updates
much easier in NLP. Exhibiting up-to-date, domain-specific knowledge is
essential for many applications: you might want to answer questions over
recent publications on COVID-19 or to develop a chatbot that guides
customers to suitable products among those currently available in a
fast-evolving inventory. For such applications, NLP models should be
able to leverage any corpus provided to them, without having to train a
new version of the model for each emerging scenario or domain.

While large language models are trained using plenty of data from the
Web, this snapshot is:

-   **Static.** The Web evolves as the world does: Wikipedia articles
    reflect new elected officials, news articles describe current events, and
    scientific papers communicate new research. Despite this, a language
    model trained in 2020 has no way to learn about 2021 events, short
    of training and releasing a new version of the model.

-   **Incomplete.** Many topics are under-represented in Web crawls like C4
    and The Pile. Suppose we seek to answer questions over the ACL
    papers published 2010–2021; there is no guarantee that The Pile
    contains all papers from the ACL Anthology a priori and there is no
    way to plug that in ad-hoc without additional training. Even when
    some ACL papers are present (e.g., through arXiv, which is included
    in The Pile), they form only a tiny sliver of the data, and it is
    difficult to reliably restrict the model to specifically those
    papers for answering NLP questions.

-   **Public-only.** Many applications hinge on private text, like internal
    company policies, in-house software documentation, copyrighted
    textbooks and novels, or personal email. Because models like GPT-3
    never see such data in their training, they are fundamentally
    incapable of exhibiting knowledge pertaining to those topics without
    special re-training or fine-tuning.

With retrieval-based NLP, models learn effective ways to encode and
extract information, allowing them to generalize to updated text,
specialized domains, or private data without resorting to additional
training. This suggests a vision where developers “plug in” their text
corpus, like in-house software documentation, which is indexed by a
powerful retrieval-based NLP model that can then answer questions, solve
classification tasks, or generate summaries using the knowledge from the
corpus, while always supporting its predictions with provenance from the
corpus.

An exciting benchmark connected to this space
is **[BEIR](https://arxiv.org/abs/2104.08663)**,
which evaluates retrievers on their capacity for search “out-of-the-box”
on unseen IR tasks, like _Argument Retrieval_, and in new domains, like
the _COVID-19 research literature_. While retrieval offers a concrete
mechanism for generalizing NLP models to new domains, not every IR model
generalizes equally: the BEIR evaluations highlight the impact of
modeling and supervision choices on generalization. For instance, due to
its late interaction modeling, a vanilla off-the-shelf ColBERT retriever
achieved the strongest recall of all competing IR models in the initial
BEIR evaluations, outperforming the other off-the-shelf dense
retrievers—namely, DPR, ANCE, SBERT, and USE-QA—on 13 out of 17
datasets. The BEIR benchmark continues to develop quickly, a recent
addition being the
**[TAS-B](https://arxiv.org/abs/2104.06967)** model,
which advances a sophisticated supervision approach to distill ColBERT
and BERT models into single-vector representations, inheriting much of
their robustness in doing so. While retrieval allows rapid deployment in new
domains, explicitly adapting retrieval to new scenarios is also
possible. This is an active area of research, with work
like **[QGen](https://arxiv.org/abs/2004.14503)** and **[AugDPR](https://arxiv.org/abs/2104.07800)** that
generate synthetic questions and use those to explicitly fine-tune
retrievers for targeting a new corpus.



### Summary: Is retrieval “all you need”?

The black-box nature of large language models like T5 and GPT-3 makes
them **inefficient** to train and deploy, **opaque** in their knowledge representations and in backing
their claims with provenance, and **static** in facing a constantly evolving world and diverse downstream contexts.
This post explores **retrieval-based NLP**, where models retrieve information
pertinent to solving their tasks from a plugged-in text corpus. This
paradigm allows NLP models to leverage the representational strengths
of language models, while needing **much smaller architectures**, offering
**transparent provenance** for claims, and enabling **efficient updates and adaptation**.

We surveyed much of the existing and emerging work in this space and
highlighted some of our work at Stanford, including
**[ColBERT](https://arxiv.org/abs/2004.12832)**
for scaling up expressive retrieval to massive corpora via late
interaction,
**[ColBERT-QA](https://arxiv.org/abs/2007.00814)** for
accurately answering open-domain questions by adapting high-recall
retrieval to the task, and
**[Baleen](https://arxiv.org/abs/2101.00436)** for
solving tasks that demand information from several independent sources
using a condensed retrieval architecture.
We continue to actively maintain
**[our code](https://github.com/stanford-futuredata/ColBERT)** as open source.



**Acknowledgments.** We would like to thank Megha Srivastava and Drew A. Hudson for helpful comments and feedback on this blog post. We also thank Ashwin Paranjape, Xiang Lisa Li, and Sidd Karamcheti for valuable and insightful discussions.



[^colbert]: Omar Khattab and Matei Zaharia. "ColBERT: Efficient and Effective Passage Search via Contextualized Late Interaction over BERT." Proceedings of the 43rd International ACM SIGIR conference on research and development in Information Retrieval. 2020.

[^colbert-qa]: Omar Khattab, Christopher Potts, Matei Zaharia; "Relevance-guided Supervision for OpenQA with ColBERT." Transactions of the Association for Computational Linguistics 2021; 9 929–944. doi: https://doi.org/10.1162/tacl_a_00405

[^baleen]: Omar Khattab, Christopher Potts, and Matei Zaharia. "Baleen: Robust Multi-Hop Reasoning at Scale via Condensed Retrieval." (To appear at NeurIPS 2021.) arXiv preprint arXiv:2101.00436 (2021).
