---
layout: post
title: "Reward Bias Substitution: Why Fixing One Reward Model Bias Can Create Another"
short-summary: "Suppose your language model has learned that longer answers receive higher rewards. You introduce a length penalty, train the model again, and successfully reduce response length. Does this solve the problem? Not quite."
summary: "Suppose your language model has learned that longer answers receive higher rewards. You introduce a length penalty, train the model again, and successfully reduce response length. Does this solve the problem? Not quite."
feature-img: "/assets/img/posts/2026-11-25-reward-bias-substitution/teaser.png"
thumbnail: "/assets/img/posts/2026-11-25-reward-bias-substitution/thumbnail.png"
author: <a href="https://www.maxlamparth.com">Max Lamparth</a>, <a href="https://www.danielfein.me">Daniel Fein</a>, <a href="https://www.andyhaupt.com">Andreas Haupt</a>, <a href="https://marcelhussing.github.io">Marcel Hussing</a>, <a href="https://mykel.kochenderfer.com">Mykel Kochenderfer</a>
tags: [reward learning, reward hacking, theory, bias mitigation, language models, AI alignment, ml, nlp, rl]
draft: True
---


{% figure %}
<img class="postimage" style="width:100%;display:block;margin:auto;padding:0 1%;" src="{{ site.baseurl }}/assets/img/posts/2026-11-25-reward-bias-substitution/teaser.png"/>
{% endfigure %}

**TL;DR:** Suppose your language model has learned that longer answers receive higher rewards. You introduce a length penalty, train the model again, and successfully reduce response length. Does this solve the problem? Not quite. We find that it makes the resulting model more confident in its answers while becoming less factually accurate. Rather than eliminating the reliance of the model on shortcuts, the mitigation changed which shortcut it exploited. We call this reward bias substitution, and it can arise whenever a learned reward signal is patched one bias at a time beyond language models. Worse, we prove that standard evaluations cannot tell a real fix from a relocated bias, because they score reward models on fixed data rather than on what the trained model actually produces. Across 30 published mitigation methods, none report the evidence needed to rule this out. We explain why this happens and give guidance for evaluating fixes so that patching one problem does not create another.

Keywords: *reward learning, reward hacking, theory, bias mitigation, language models, AI alignment*

Based on the publication available on [OpenReview](https://openreview.net/forum?id=Auv9cgHOSa) (NeurIPS 2026) and as pre-print on [arXiv](https://arxiv.org/abs/2605.27996). 

## Learned rewards cannot be specified in one shot
{: style="margin-top:1.5em;"}

For many tasks, we cannot automatically verify the output quality of an AI system. In the case of language models, nothing can objectively tell us whether a response was helpful, honest, or well argued. In contrast, we can check the correctness of math problems or code with test cases. In *non-verifiable domains*, we learn a reward from human judgments as a proxy for the unknown true reward. However, surface features such as response length or background in generated images can correlate with quality without always being meaningful indicators of it.  Preference data alone cannot separate the informative part from the spurious one. 

Learned rewards inherit these spurious correlations, which surface primarily when you use the reward to train a model (a “policy”). Thus, learned rewards have to be improved in context of the model you want to train, by not only diagnosing a bias and correcting the reward, but also looking at the following re-training with the de-biased reward. Meaning, correcting a reward model bias is only half the job, as we also need to examine how the policy behaves after training with the corrected reward. We argue this iterative approach is unavoidable in non-verifiable domains, with current single bias mitigations as the basic iteration step. Our paper asks how we can tell if that step worked as intended.

We study the standard setup in which a model is trained to score as high as possible under the learned reward while being penalized for drifting too far from the model it started as (KL-regularized reinforcement learning against a reference policy). Common methods, especially for language models, include Reinforcement Learning from Human Feedback (RLHF), Direct Preference Optimization (DPO), and extensions of these methods. Our framework applies to any scalar proxy reward (the final reward is a single number), including Bradley-Terry, generative, and rubric-scored, and to the implicit rewards of DPO-like methods.

For language models, learned rewards commonly pick up spurious correlations with length, sycophancy, confidence, or style. Length is the best known, leading to overly verbose policies. Many mitigations target these features directly, for instance by separating length from quality inside the reward model or adding length penalties as reward shaping during training.

Two properties of this setup explain why these biases are learned and why they are hard to remove. First, preference data constrains the reward only through comparisons between responses to the same prompt, so many reward functions fit the same data equally well and a spurious feature can be indistinguishable from a relevant one. Second, the trained policy responds not only to which response scores higher but to the reward values themselves, including their scale, so two rewards that agree on every comparison can still produce different policies.

## Removing a feature from the reward does not remove the pressure to exploit it
{: style="margin-top:1.5em;"}

Spurious features are partially informative about quality and correlated with one another, so suppressing one redirects optimization onto the rest, similar to shortcuts in supervised learning[^li2023]. We call this **reward bias substitution**. It is not specific to length or to RLHF, but a general type of reward hacking in which a single-axis mitigation relocates optimization pressure onto correlated proxies.

{% figure %}
<img class="postimage" style="width:100%;display:block;margin:auto;padding:0 6%;" src="{{ site.baseurl }}/assets/img/posts/2026-11-25-reward-bias-substitution/fig1.png"/>
<figcaption style="color:#777;">
<b>Figure 1</b>. A length penalty during standard training with GRPO shortens responses (left), but optimization pressure is redirected onto verbalized confidence (middle), and free-form factual accuracy drops while multiple-choice accuracy holds (right). Confidence and accuracy stay at base level without the penalty, so the penalty, not RLHF, causes the shift. Three penalty strengths, four seeds each, 95 percent bootstrapped confidence intervals.
</figcaption>
{% endfigure %}

Let’s look at an example! Figure 1 shows a case of reward bias substitution and how this failure happens in standard RLHF. We train the Llama-3.2-3B-Instruct model using GRPO against the Skywork-Reward-V2-Llama-3.1-8B reward model on the UltraFeedback data set and subtract a fixed penalty per 100 response tokens, scaled with a factor λ (λ = 0 means training without length penalty). For rising λ, mean response length falls as intended, and MMLU accuracy is unchanged. Yet verbalized confidence rises, confidence-correctness AUROC drops, and free-form accuracy on TriviaQA falls. Runs without the penalty leave all three at base level, so the penalty causes the degradation, not doing RLHF itself. Penalizing length discourages hedging, which reward models score as confidence, so the freed pressure lands on confidence while factual accuracy drops.

{% figure %}
<img class="postimage" style="width:100%;display:block;margin:auto;padding:0 6%;" src="{{ site.baseurl }}/assets/img/posts/2026-11-25-reward-bias-substitution/fig2.png"/>
<figcaption style="color:#777;">
<b>Figure 2</b>. Schematic of reward bias substitution targeting length bias in language models. Length and confidence are correlated spurious features, as is also indicated by the tilt of the starting model (gray). The learned reward (orange) pulls the trained model (purple) towards longer outputs before mitigation (left panel). A length-targeting mitigation (middle panel) with an audit data set (green) that controls for quality and is sensitive to length, cuts the length reliance but the optimization pressure is redirected onto confidence. The rerouting is invisible on the audit data and shows up only in the outputs of the trained model. Overall model quality decreases even though the length bias appears removed (right panel).
</figcaption>
{% endfigure %}

Such failures can go unnoticed because a bias mitigation is commonly tested and used on different data. A mitigation is built by estimating the reliance of the reward on the target feature on an evaluation or calibration data set (*audit distribution* in green, Figure 2). On this data set, the reliance of a reward on a bias can be reduced to zero by construction. However, the policy produces its own data set through its outputs (*policy-induced distribution* in purple, Figure 2), where the mitigated reward can still depend on the targeted feature and, as in Figure 2, now pulls the model toward correlated features instead. The mitigation is tuned to the audit data, but training responds to how the reward behaves on the model’s own outputs. We call this the **measurement-versus-optimization gap**. 

Versions of this gap are known for a fixed reward model[^fluri2025][^wen2025][^gleave2021], but here the mitigation itself creates it. Existing definitions of reward hacking[^skalse2022][^laidlaw2025][^kwa2024] are stated for a given reward model, not for an operator editing it, so they cannot separate removing a bias from relocating it. And on the audit distribution, a successful mitigation, a relocation of the bias, and an overcorrection that lowers true reward can all produce the same scores.

## A regime taxonomy for single-axis mitigations
{: style="margin-top:1.5em;"}

We classify the outcome of a mitigation along two dimensions, the change in model quality of the trained model and whether optimization pressure redirects onto another spurious feature, measured by how much the trained model outputs shift with respect to tracked features.

| Model quality | No redirection | Redirection onto another spurious feature |
|---|---|---|
| Goes up | **R0** Successful mitigation | **R0cont** Contaminated success |
| Unchanged | **R3** Silent non-op | **R1** Bias substitution (neutral) |
| Goes down | **R2** Overcorrection | **R1** Bias substitution (harmful) |

Besides bias substitution, overcorrection (R2) arises from a scale overshoot, fixable by rescaling, or from removing a genuinely informative part of a partially informative feature. Contaminated success (R0cont) is the outcome audits most easily mistake for R0, since the gain from reducing the targeted feature outweighs the cost of the redirection, but can also be an intentional trade-off. A fifth regime, audit-distribution sensitivity (R4), covers mitigations whose regime changes with the audit set choice, for instance how the same mitigation can produce different outcomes depending on whether human or LLM-judges annotated the audit data.

## What can certify a mitigation
{: style="margin-top:1.5em;"}

No test computed on audit data alone, whatever score it uses, can tell R0, R0cont, R1, and R2 apart in general, even if the test had access to the true reward. We prove this by constructing cases that look identical on the audit data but end up in each of the four regimes, differing only in the starting model. Looking at the outputs of the trained model before and after mitigation, and tracking how other features shift, is enough to tell them apart. A paper claiming successful mitigation should therefore evaluate at the policy-induced distribution (for instance Best-of-N sampling or a short RL run from a fixed starting model), track at least length, confidence, and sycophancy there, report the reward scale before and after mitigation, and validate across at least two reference policies. We also give prescriptions for benchmark developers.

## Bias substitution in the wild
{: style="margin-top:1.5em;"}

Re-evaluating two published length-debiasing operators across five reward models under Best-of-N, one operator removes the overall correlation between reward and length on AlpacaEval, yet when comparing responses to the same prompt the reward now prefers shorter answers and GSM8K Best-of-N accuracy drops in one setting. Across 30 published mitigation works, none report the evidence needed to certify successful mitigation, because they either do not evaluate on the outputs of trained models, or do not track how other features change. In our own analysis and survey of prior findings, we find that the preconditions for substitution hold in practice. Analyzing the responses from LLMs to r/AITA posts from[^cheng2026], we find that  sycophantic responses are shorter after controlling for model and prompt, showing that sycophancy labels and response length can be coupled.

## Outlook
{: style="margin-top:1.5em;"}

Current metrics cannot distinguish a learned reward that became less hackable from one that shifted the exploited feature, and that ambiguity carries over to any claim that a model trained with the corrected reward is less sycophantic, more honest, or better calibrated. Because learned rewards (including generative reward models with rubrics) on non-verifiable objectives cannot be specified in one shot and have to be improved iteratively, each uncertified iteration can relocate a bias to the next round. Our checklist certifies "no substitution within the measured features," not "no substitution." For language models, length is the central axis through which sycophancy and confidence couple, and mapping the rest is the direction we most want pursued. So any new method targeting length needs to also track confidence, calibration, and sycophancy flip rates, while methods targeting any of the others must at least track length.

## How to Cite
{: style="margin-top:1.5em;"}

This blog post is based on the paper *Reward Bias Substitution: Single-Axis Bias Mitigations Redirect Optimization Pressure* and can be cited as

```
@inproceedings{lamparth2026reward,
title={Reward Bias Substitution: Single-Axis Bias Mitigations Redirect Optimization Pressure},
author={Max Lamparth and Daniel Fein and Andreas Haupt and Marcel Hussing and Mykel Kochenderfer},
booktitle={The Fortieth Annual Conference on Neural Information Processing Systems},
year={2026},
url={https://openreview.net/forum?id=Auv9cgHOSa}
}
```

or 


> Max Lamparth, Daniel Fein, Andreas Haupt, Marcel Hussing, and Mykel Kochenderfer. Reward Bias Substitution: Single-Axis Bias Mitigations Redirect Optimization Pressure. In The Fortieth Annual Conference on Neural Information Processing Systems, 2026. URL <https://openreview.net/forum?id=Auv9cgHOSa>.

[^li2023]: Zhiheng Li, Ivan Evtimov, Albert Gordo, Caner Hazirbas, Tal Hassner, Cristian Canton Ferrer, Chenliang Xu, and Mark Ibrahim. A whac-a-mole dilemma: Shortcuts come in multiples where mitigating one amplifies others. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2023.
[^fluri2025]: Lukas Fluri, Leon Lang, Alessandro Abate, Patrick Forré, David Krueger, and Joar Max Viktor Skalse. The perils of optimizing learned reward functions: Low training error does not guarantee low regret. In Proceedings of the 42nd International Conference on Machine Learning, volume 267 of Proceedings of Machine Learning Research, pages 17306–17377. PMLR, 13–19 Jul 2025. URL <https://proceedings.mlr.press/v267/fluri25a.html>.
[^wen2025]: Xueru Wen, Jie Lou, Yaojie Lu, Hongyu Lin, Xing Yu, Xinyu Lu, Ben He, Xianpei Han, Debing Zhang, and Le Sun. Rethinking reward model evaluation: Are we barking up the wrong tree? In The Thirteenth International Conference on Learning Representations, 2025. URL <https://openreview.net/forum?id=Cnwz9jONi5>.
[^gleave2021]: Adam Gleave, Michael D Dennis, Shane Legg, Stuart Russell, and Jan Leike. Quantifying differences in reward functions. In International Conference on Learning Representations, 2021. URL <https://openreview.net/forum?id=LwEQnp6CYev>.
[^skalse2022]: Joar Skalse, Nikolaus H. R. Howe, Dmitrii Krasheninnikov, and David Krueger. Defining and characterizing reward hacking. In Proceedings of the 36th International Conference on Neural Information Processing Systems, NIPS ’22, Red Hook, NY, USA, 2022. Curran Associates Inc.
[^laidlaw2025]: Cassidy Laidlaw, Shivam Singhal, and Anca Dragan. Correlated proxies: A new definition and improved mitigation for reward hacking. In The Thirteenth International Conference on Learning Representations, 2025. URL <https://openreview.net/forum?id=msEr27EejF>.
[^kwa2024]: Thomas Kwa, Drake Thomas, and Adrià Garriga-Alonso. Catastrophic goodhart: regularizing RLHF with KL divergence does not mitigate heavy-tailed reward misspecification. In The Thirty-eighth Annual Conference on Neural Information Processing Systems, 2024. URL <https://openreview.net/forum?id=UXuBzWoZGK>.
[^cheng2026]: Myra Cheng, Cinoo Lee, Pranav Khadpe, Sunny Yu, Dyllan Han, and Dan Jurafsky. Sycophantic AI decreases prosocial intentions and promotes dependence. Science, 391:eaec8352, 2026. doi: [10.1126/science.aec8352](https://doi.org/10.1126/science.aec8352).