---
layout: post
title: "What Matters in Learning from Offline Human Demonstrations for Robot Manipulation"
short-summary: "We conducted an extensive study of six offline learning algorithms for robot manipulation on five simulated and three real-world multi-stage manipulation tasks of varying complexity, and with datasets of varying quality. Our study analyzes the most critical challenges when learning from offline human data for manipulation."
summary: "We conducted an extensive study of six offline learning algorithms for robot manipulation on five simulated and three real-world multi-stage manipulation tasks of varying complexity, and with datasets of varying quality. Our study analyzes the most critical challenges when learning from offline human data for manipulation."
feature-img: "assets/img/posts/2021-08-08-robomimic/thumbnail.png"
thumbnail: "assets/img/posts/2021-08-08-robomimic/thumbnail.png"
author: <a href="http://web.stanford.edu/~amandlek/">Ajay Mandlekar</a>
tags: [imitation learning, reinforcement learning, rl, ml, robotics]
---

## Overview

Imitation Learning is a promising approach to endow robots with various complex manipulation capabilities. By allowing robots to learn from datasets collected by humans, robots can learn to perform the same skills that were demonstrated by the human. Typically, these datasets are collected by having humans control robot arms, guiding them through different tasks. While this paradigm has proved effective, a lack of open-source human datasets and reproducible learning methods make assessing the state of the field difficult. In this paper, we conduct an extensive study of six offline learning algorithms for robot manipulation on five simulated and three real-world multi-stage manipulation tasks of varying complexity, and with datasets of varying quality. Our study analyzes the most critical challenges when learning from offline human data for manipulation. 

Based on the study, we derive several lessons to understand the challenges in learning from human demonstrations, including the sensitivity to different algorithmic design choices, the dependence on the quality of the demonstrations, and the variability based on the stopping criteria due to the different objectives in training and evaluation. We also highlight opportunities for learning from human datasets, such as the ability to learn proficient policies on challenging, multi-stage tasks beyond the scope of current reinforcement learning methods, and the ability to easily scale to natural, real-world manipulation scenarios where only raw sensory signals are available. 

**We have open-sourced our datasets and all algorithm implementations to facilitate future research and fair comparisons in learning from human demonstration data.** Please see the [robomimic website](https://arise-initiative.github.io/robomimic-web/) for more information.


{% figure %}
<img src="{{ site.baseurl }}/assets/img/posts/2021-08-08-robomimic/overview.png" class="postimagetwothird"/>
<figcaption>
In this study, we investigate several challenges of offline learning from human datasets and extract lessons to guide future work.
</figcaption>
{% endfigure %}

## Why is learning from human-labeled datasets difficult?

We explore five challenges in learning from human-labeled datasets.

{% figure %}
<img src="{{ site.baseurl }}/assets/img/posts/2021-08-08-robomimic/challenges.png" class="postimagetwothird"/>
{% endfigure %}

- **(C1) Unobserved Factors in Human Decision Making.** Humans are not perfect Markovian agents. In addition to what they currently see, their actions may be influenced by other external factors - such as the device they are using to control the robot and the history of the actions that they have provided.
<br><br>
- **(C2) Mixed Demonstration Quality.** Collecting data from multiple humans can result in mixed quality data, since some people might be better quality supervisors than others. 
<br><br>
- **(C3) Dependence on dataset size.** When a robot learns from an offline dataset, it needs to understand how it should act (action) in every scenario that it might encounter (state). This is why the coverage of states and actions in the dataset matters. Larger datasets are likely to contain more situations, and are therefore likely to train better robots.
<br><br>
- **(C4) Train Objective ≠ Eval Objective.** Unlike traditional supervised learning, where validation loss is a strong indicator of how good a model is, policies are usually trained with surrogate losses. Consider an example where we train a policy via Behavioral Cloning from a set of demonstrations on a block lifting task. Here, the policy is trained to replicate the actions taken by the demonstrator, but this is not necessarily equivalent to optimizing the block lifting success rate (see [the Dagger paper](https://arxiv.org/abs/1011.0686) for a more precise explanation). This makes it hard to know which trained policy checkpoints are good without trying out each and every model directly on the robot -- a time consuming process.
<br><br>
- **(C5) Sensitivity to Agent Design Decisions.** Performance can be very sensitive to important agent design decisions, like the observation space and hyperparameters used for learning.


## Study Design

In this section, we summarize the tasks (5 simulated and 3 real), datasets (3 different variants), algorithms (6 offline methods, including 3 imitation and 3 batch reinforcement), and observation spaces (2 main variants) that we explored in our study.

### Tasks


{% figure %}

<figure class="postfigurequarter">
  <video autoplay loop muted playsinline controls class="postimage_unpadded">
    <source src="{{ site.baseurl }}/assets/img/posts/2021-08-08-robomimic/task_lift.mp4" type="video/mp4">
  </video>
  <figcaption>
  Lift
  </figcaption>
</figure>

<figure class="postfigurequarter">
  <video autoplay loop muted playsinline controls class="postimage_unpadded">
    <source src="{{ site.baseurl }}/assets/img/posts/2021-08-08-robomimic/task_can.mp4" type="video/mp4">
  </video>
  <figcaption>
  Can
  </figcaption>
</figure>

<figure class="postfigurequarter">
  <video autoplay loop muted playsinline controls class="postimage_unpadded">
    <source src="{{ site.baseurl }}/assets/img/posts/2021-08-08-robomimic/task_tool_hang.mp4" type="video/mp4">
  </video>
  <figcaption>
  Tool Hang
  </figcaption>
</figure>

<figure class="postfigurequarter">
  <video autoplay loop muted playsinline controls class="postimage_unpadded">
    <source src="{{ site.baseurl }}/assets/img/posts/2021-08-08-robomimic/task_square.mp4" type="video/mp4">
  </video>
  <figcaption>
  Square
  </figcaption>
</figure>

<figure class="postfigurequarter">
  <video autoplay loop muted playsinline controls class="postimage_unpadded">
    <source src="{{ site.baseurl }}/assets/img/posts/2021-08-08-robomimic/task_lift_real.mp4" type="video/mp4">
  </video>
  <figcaption>
  Lift (Real)
  </figcaption>
</figure>

<figure class="postfigurequarter">
  <video autoplay loop muted playsinline controls class="postimage_unpadded">
    <source src="{{ site.baseurl }}/assets/img/posts/2021-08-08-robomimic/task_can_real.mp4" type="video/mp4">
  </video>
  <figcaption>
  Can (Real)
  </figcaption>
</figure>

<figure class="postfigurequarter">
  <video autoplay loop muted playsinline controls class="postimage_unpadded">
    <source src="{{ site.baseurl }}/assets/img/posts/2021-08-08-robomimic/task_tool_hang_real.mp4" type="video/mp4">
  </video>
  <figcaption>
  Tool Hang (Real)
  </figcaption>
</figure>

<figure class="postfigurequarter">
  <video autoplay loop muted playsinline controls class="postimage_unpadded">
    <source src="{{ site.baseurl }}/assets/img/posts/2021-08-08-robomimic/task_transport.mp4" type="video/mp4">
  </video>
  <figcaption>
  Transport
  </figcaption>
</figure>

<figcaption>
We collect datasets across 6 operators of varying proficiency and evaluate offline policy learning methods on 8 challenging manipulation tasks that test a wide range of manipulation capabilities including pick-and-place, multi-arm coordination, and high-precision insertion and assembly.
</figcaption>

{% endfigure %}

### Task Reset Distributions

When measuring the task success rate of a policy, the policy is evaluated across several trials. At the start of each trial, the initial placement of all objects in the task are randomized from a task reset distribution. The videos below show this distribution for each task. This gives an impression of the range of different scenarios that a trained policy is supposed to be able to handle.

{% figure %}
<figure class="postfigurequarter">
  <img src="{{ site.baseurl }}/assets/img/posts/2021-08-08-robomimic/reset_lift.gif" class="postimage_unpadded"/>
  <figcaption>
  Lift
  </figcaption>
</figure>

<figure class="postfigurequarter">
  <img src="{{ site.baseurl }}/assets/img/posts/2021-08-08-robomimic/reset_can.gif" class="postimage_unpadded"/>
  <figcaption>
  Can
  </figcaption>
</figure>

<figure class="postfigurequarter">
  <img src="{{ site.baseurl }}/assets/img/posts/2021-08-08-robomimic/reset_tool_hang.gif" class="postimage_unpadded"/>
  <figcaption>
  Tool Hang
  </figcaption>
</figure>

<figure class="postfigurequarter">
  <img src="{{ site.baseurl }}/assets/img/posts/2021-08-08-robomimic/reset_square.gif" class="postimage_unpadded"/>
  <figcaption>
  Square
  </figcaption>
</figure>

<figure class="postfigurequarter">
  <img src="{{ site.baseurl }}/assets/img/posts/2021-08-08-robomimic/reset_lift_real.gif" class="postimage_unpadded"/>
  <figcaption>
  Lift (Real)
  </figcaption>
</figure>

<figure class="postfigurequarter">
  <img src="{{ site.baseurl }}/assets/img/posts/2021-08-08-robomimic/reset_can_real.gif" class="postimage_unpadded"/>
  <figcaption>
  Can (Real)
  </figcaption>
</figure>

<figure class="postfigurequarter">
  <img src="{{ site.baseurl }}/assets/img/posts/2021-08-08-robomimic/reset_tool_hang_real.gif" class="postimage_unpadded"/>
  <figcaption>
  Tool Hang (Real)
  </figcaption>
</figure>

<figure class="postfigurequarter">
  <img src="{{ site.baseurl }}/assets/img/posts/2021-08-08-robomimic/reset_transport.gif" class="postimage_unpadded"/>
  <figcaption>
  Transport
  </figcaption>
</figure>

<figcaption>
We show the task reset distributions for each task, which governs the initial placement of all objects in the scene at the start of each episode. Initial states are sampled from this distribution at both train and evaluation time.
</figcaption>

{% endfigure %}


### Datasets

{% figure %}
<img src="{{ site.baseurl }}/assets/img/posts/2021-08-08-robomimic/dataset_overview.png" class="postimagefourfifth"/>
<figcaption>
We collected 3 kinds of datasets in this study.
</figcaption>
{% endfigure %}

#### Machine-Generated

These datasets consist of rollouts from a series of [SAC](https://arxiv.org/abs/1801.01290) agent checkpoints trained on Lift and Can, instead of humans. As a result, they contain random, suboptimal, and expert data due to the varied success rates of the agents that generated the data. This kind of mixed quality data is common in offline RL works (e.g. [D4RL](https://github.com/rail-berkeley/d4rl/tree/master/d4rl), [RLUnplugged](https://github.com/deepmind/deepmind-research/tree/master/rl_unplugged)).


{% figure %}

<figure class="postfigurethird">
  <video autoplay loop muted playsinline controls class="postimage_unpadded">
    <source src="{{ site.baseurl }}/assets/img/posts/2021-08-08-robomimic/playback_lift_rb_1.5k.mp4" type="video/mp4">
  </video>
  <figcaption>
  Lift (MG)
  </figcaption>
</figure>

<figure class="postfigurethird">
  <video autoplay loop muted playsinline controls class="postimage_unpadded">
    <source src="{{ site.baseurl }}/assets/img/posts/2021-08-08-robomimic/playback_can_rb_3.9k.mp4" type="video/mp4">
  </video>
  <figcaption>
  Can (MG)
  </figcaption>
</figure>

<figcaption>
Lift and Can Machine-Generated datasets.
</figcaption>

{% endfigure %}


#### Proficient-Human

These datasets consist of 200 demonstrations collected from a single proficient human operator using [RoboTurk](https://roboturk.stanford.edu/).

{% figure %}

<figure class="postfigurethird">
  <video autoplay loop muted playsinline controls class="postimage_unpadded">
    <source src="{{ site.baseurl }}/assets/img/posts/2021-08-08-robomimic/playback_lift_se.mp4" type="video/mp4">
  </video>
  <figcaption>
  Lift (PH)
  </figcaption>
</figure>

<figure class="postfigurethird">
  <video autoplay loop muted playsinline controls class="postimage_unpadded">
    <source src="{{ site.baseurl }}/assets/img/posts/2021-08-08-robomimic/playback_can_se.mp4" type="video/mp4">
  </video>
  <figcaption>
  Can (PH)
  </figcaption>
</figure>

<figure class="postfigurethird">
  <video autoplay loop muted playsinline controls class="postimage_unpadded">
    <source src="{{ site.baseurl }}/assets/img/posts/2021-08-08-robomimic/playback_square_se.mp4" type="video/mp4">
  </video>
  <figcaption>
  Square (PH)
  </figcaption>
</figure>

<figure class="postfigurethird">
  <video autoplay loop muted playsinline controls class="postimage_unpadded">
    <source src="{{ site.baseurl }}/assets/img/posts/2021-08-08-robomimic/playback_transport_se_trim.mp4" type="video/mp4">
  </video>
  <figcaption>
  Transport (PH)
  </figcaption>
</figure>

<figure class="postfigurethird">
  <video autoplay loop muted playsinline controls class="postimage_unpadded">
    <source src="{{ site.baseurl }}/assets/img/posts/2021-08-08-robomimic/playback_tool_hang_se_trim.mp4" type="video/mp4">
  </video>
  <figcaption>
  Tool Hang (PH)
  </figcaption>
</figure>

<figcaption>
Proficient-Human datasets generated by 1 proficient operator (with the exception of Transport, which had 2 proficient operators working together).
</figcaption>

{% endfigure %}


#### Multi-Human

These datasets consist of 300 demonstrations collected from six human operators of varied proficiency using [RoboTurk](https://roboturk.stanford.edu/). Each operator falls into one of 3 groups - "Worse", "Okay", and "Better" -- each group contains two operators. Each operator collected 50 demonstrations per task. As a result, these datasets contain mixed quality human demonstration data. We show videos for a single operator from each group. 


{% figure %}

<figure class="postfigurethird">
  <video autoplay loop muted playsinline controls class="postimage_unpadded">
    <source src="{{ site.baseurl }}/assets/img/posts/2021-08-08-robomimic/playback_lift_mh_worse_1.mp4" type="video/mp4">
  </video>
  <figcaption>
  Lift (MH) - Worse
  </figcaption>
</figure>

<figure class="postfigurethird">
  <video autoplay loop muted playsinline controls class="postimage_unpadded">
    <source src="{{ site.baseurl }}/assets/img/posts/2021-08-08-robomimic/playback_lift_mh_okay_1.mp4" type="video/mp4">
  </video>
  <figcaption>
  Lift (MH) - Okay
  </figcaption>
</figure>

<figure class="postfigurethird">
  <video autoplay loop muted playsinline controls class="postimage_unpadded">
    <source src="{{ site.baseurl }}/assets/img/posts/2021-08-08-robomimic/playback_lift_mh_better_1.mp4" type="video/mp4">
  </video>
  <figcaption>
  Lift (MH) - Better
  </figcaption>
</figure>

<figcaption>
Multi-Human Lift dataset. The videos show three operators - one that's "worse" (left), "okay" (middle) and "better" (right).
</figcaption>

{% endfigure %}


{% figure %}

<figure class="postfigurethird">
  <video autoplay loop muted playsinline controls class="postimage_unpadded">
    <source src="{{ site.baseurl }}/assets/img/posts/2021-08-08-robomimic/playback_can_mh_worse_1_trim.mp4" type="video/mp4">
  </video>
  <figcaption>
  Can (MH) - Worse
  </figcaption>
</figure>

<figure class="postfigurethird">
  <video autoplay loop muted playsinline controls class="postimage_unpadded">
    <source src="{{ site.baseurl }}/assets/img/posts/2021-08-08-robomimic/playback_can_mh_okay_1_trim.mp4" type="video/mp4">
  </video>
  <figcaption>
  Can (MH) - Okay
  </figcaption>
</figure>

<figure class="postfigurethird">
  <video autoplay loop muted playsinline controls class="postimage_unpadded">
    <source src="{{ site.baseurl }}/assets/img/posts/2021-08-08-robomimic/playback_can_mh_better_1.mp4" type="video/mp4">
  </video>
  <figcaption>
  Can (MH) - Better
  </figcaption>
</figure>

<figcaption>
Multi-Human Can dataset. The videos show three operators - one that's "worse" (left), "okay" (middle) and "better" (right).
</figcaption>

{% endfigure %}


{% figure %}

<figure class="postfigurethird">
  <video autoplay loop muted playsinline controls class="postimage_unpadded">
    <source src="{{ site.baseurl }}/assets/img/posts/2021-08-08-robomimic/playback_square_mh_worse_1_trim.mp4" type="video/mp4">
  </video>
  <figcaption>
  Square (MH) - Worse
  </figcaption>
</figure>

<figure class="postfigurethird">
  <video autoplay loop muted playsinline controls class="postimage_unpadded">
    <source src="{{ site.baseurl }}/assets/img/posts/2021-08-08-robomimic/playback_square_mh_okay_1_trim.mp4" type="video/mp4">
  </video>
  <figcaption>
  Square (MH) - Okay
  </figcaption>
</figure>

<figure class="postfigurethird">
  <video autoplay loop muted playsinline controls class="postimage_unpadded">
    <source src="{{ site.baseurl }}/assets/img/posts/2021-08-08-robomimic/playback_square_mh_better_1_trim.mp4" type="video/mp4">
  </video>
  <figcaption>
  Square (MH) - Better
  </figcaption>
</figure>

<figcaption>
Multi-Human Square dataset. The videos show three operators - one that's "worse" (left), "okay" (middle) and "better" (right).
</figcaption>

{% endfigure %}


{% figure %}

<figure class="postfigurethird">
  <video autoplay loop muted playsinline controls class="postimage_unpadded">
    <source src="{{ site.baseurl }}/assets/img/posts/2021-08-08-robomimic/playback_transport_mh_worse_trim.mp4" type="video/mp4">
  </video>
  <figcaption>
  Transport (MH) - Worse-Worse
  </figcaption>
</figure>

<figure class="postfigurethird">
  <video autoplay loop muted playsinline controls class="postimage_unpadded">
    <source src="{{ site.baseurl }}/assets/img/posts/2021-08-08-robomimic/playback_transport_mh_okay_trim.mp4" type="video/mp4">
  </video>
  <figcaption>
  Transport (MH) - Okay-Okay
  </figcaption>
</figure>

<figure class="postfigurethird">
  <video autoplay loop muted playsinline controls class="postimage_unpadded">
    <source src="{{ site.baseurl }}/assets/img/posts/2021-08-08-robomimic/playback_transport_mh_better_trim.mp4" type="video/mp4">
  </video>
  <figcaption>
  Transport (MH) - Better-Better
  </figcaption>
</figure>

<figure class="postfigurethird">
  <video autoplay loop muted playsinline controls class="postimage_unpadded">
    <source src="{{ site.baseurl }}/assets/img/posts/2021-08-08-robomimic/playback_transport_mh_worse_okay_trim.mp4" type="video/mp4">
  </video>
  <figcaption>
  Transport (MH) - Worse-Okay
  </figcaption>
</figure>

<figure class="postfigurethird">
  <video autoplay loop muted playsinline controls class="postimage_unpadded">
    <source src="{{ site.baseurl }}/assets/img/posts/2021-08-08-robomimic/playback_transport_mh_worse_better_trim.mp4" type="video/mp4">
  </video>
  <figcaption>
  Transport (MH) - Worse-Better
  </figcaption>
</figure>

<figure class="postfigurethird">
  <video autoplay loop muted playsinline controls class="postimage_unpadded">
    <source src="{{ site.baseurl }}/assets/img/posts/2021-08-08-robomimic/playback_transport_mh_okay_better_trim.mp4" type="video/mp4">
  </video>
  <figcaption>
  Transport (MH) - Okay-Better
  </figcaption>
</figure>

<figcaption>
Multi-Human Transport dataset. These were collected using pairs of operators with <a href="https://arxiv.org/abs/2012.06738">Multi-Arm RoboTurk</a> (each one controlled 1 robot arm). We collected 50 demonstrations per combination of the operator subgroups.
</figcaption>

{% endfigure %}


### Algorithms

We evaluated 6 different offline learning algorithms in this study, including 3 imitation learning and 3 batch (offline) reinforcement learning algorithms.

{% figure %}
<img src="{{ site.baseurl }}/assets/img/posts/2021-08-08-robomimic/algo_overview.png" class="postimagefourfifth"/>
<figcaption>
We evaluated 6 different offline learning algorithms in this study, including 3 imitation learning and 3 batch (offline) reinforcement learning algorithms.
</figcaption>
{% endfigure %}

- **BC**: standard Behavioral Cloning, which is direct regression from observations to actions.
- **BC-RNN**: Behavioral Cloning with a policy network that's a recurrent neural network (RNN), which allows modeling temporal correlations in decision-making.
- **HBC**: Hierarchical Behavioral Cloning, where a high-level subgoal planner is trained to predict future observations, and a low-level recurrent policy is conditioned on a future observation (subgoal) to predict action sequences (see [Mandlekar\*, Xu\* et al. (2020)](https://arxiv.org/abs/2003.06085) and [Tung\*, Wong\* et al. (2021)](https://arxiv.org/abs/2012.06738) for more details).
- **BCQ**: Batch-Constrained Q-Learning, a batch reinforcement learning method proposed in [Fujimoto et al. (2019)](https://arxiv.org/abs/1812.02900).
- **CQL**: Conservative Q-Learning, a batch reinforcement learning method proposed in [Kumar et al. (2020)](https://arxiv.org/abs/2006.04779).
- **IRIS**: Implicit Reinforcement without Interaction, a batch reinforcement learning method proposed in [Mandlekar et al. (2020)](https://arxiv.org/abs/1911.05321).


### Observation Spaces

We study two different observation spaces in this work -- low-dimensional observations and image observations.

{% figure %}
<img src="{{ site.baseurl }}/assets/img/posts/2021-08-08-robomimic/obs_overview.png" class="postimagefourfifth"/>
<figcaption>
We study two different observation spaces in this work.
</figcaption>
{% endfigure %}


#### Image Observations

We provide examples of the image observations used in each task below.

{% figure %}

<figure class="postfigurehalf">
  <video autoplay loop muted playsinline controls class="postimage_unpadded">
    <source src="{{ site.baseurl }}/assets/img/posts/2021-08-08-robomimic/obs_can_trim.mp4" type="video/mp4">
  </video>
  <figcaption>
  Most tasks have a front view and wrist view camera. The front view matches the view provided to the operator during data collection.
  </figcaption>
</figure>

<figure class="postfigurehalf">
  <video autoplay loop muted playsinline controls class="postimage_unpadded">
    <source src="{{ site.baseurl }}/assets/img/posts/2021-08-08-robomimic/obs_tool_hang_trim.mp4" type="video/mp4">
  </video>
  <figcaption>
  Tool Hang has a side view and wrist view camera. The side view matches the view provided to the operator during data collection.
  </figcaption>
</figure>

<figure class="postfigurefourfifths">
  <video autoplay loop muted playsinline controls class="postimage_unpadded">
    <source src="{{ site.baseurl }}/assets/img/posts/2021-08-08-robomimic/obs_transport_trim.mp4" type="video/mp4">
  </video>
  <figcaption>
  Transport has a shoulder view and wrist view camera per arm. The shoulder view cameras match the views provided to each operator during data collection.
  </figcaption>
</figure>

{% endfigure %}


## Summary of Lessons Learned

In this section, we briefly highlight the lessons we learned from our study. See the paper for more thorough results and discussion.

### Lesson 1: History-dependent models are extremely effective.

We found that there is a substantial performance gap between BC-RNN and BC, which highlights the benefits of history-dependence. This performance gap is larger for longer-horizon tasks (e.g. \~55% for the Transport (PH) dataset compared to \~5% for the Square (PH) dataset)) and also larger for multi-human data compared to single-human data (e.g.\~25% for Square (MH) compared to \~5% for Square (PH)). 

{% figure %}
<img src="{{ site.baseurl }}/assets/img/posts/2021-08-08-robomimic/lesson_1.png" class="postimagefourfifth"/>
<figcaption>
Methods that make decisions based on history, such as BC-RNN and HBC, outperform other methods on human datasets.
</figcaption>
{% endfigure %}

### Lesson 2: Batch (Offline) RL struggles with suboptimal human data.

Recent batch (offline) RL algorithms such as BCQ and CQL have demonstrated excellent results in learning from suboptimal and multi-modal machine-generated datasets. Our results confirm the capacity of such algorithms to work well – BCQ in particular performs strongly on our agent-generated MG datasets that consist of a diverse mixture of good and poor policies (for example, BCQ achieves 91.3% success rate on Lift (MG) compared to BC which achieves 65.3%).

Surprisingly though, neither BCQ nor CQL performs particularly well on these human-generated datasets. For example, BCQ and CQL achieve 62.7% and 22.0% success respectively on the Can (MH) dataset, compared to BC-RNN which achieves 100% success. This puts the ability of such algorithms to learn from more natural dataset distributions into question (instead of those collected via RL exploration or pre-trained agents). There is an opportunity for future work in batch RL to resolve this gap.

{% figure %}
<img src="{{ site.baseurl }}/assets/img/posts/2021-08-08-robomimic/lesson_2.png" class="postimagefourfifth"/>
<figcaption>
While batch (offline) RL methods are proficient at dealing with mixed quality machine-generated data, they struggle to deal with mixed quality human data.
</figcaption>
{% endfigure %}

{% figure %}
<video autoplay loop muted playsinline controls class="postimagefourfifth">
  <source src="{{ site.baseurl }}/assets/img/posts/2021-08-08-robomimic/can_paired.mp4" type="video/mp4">
</video>
<figcaption>
To further evaluate methods in a simpler setting, we collected the Can Paired dataset, where every task instance has two demonstrations, one success and one failure. Even this simple setting, where each start state has exactly one positive and one negative demonstration, poses a problem.
</figcaption>
{% endfigure %}

### Lesson 3: Improving offline policy selection is important.

{% figure %}
<img src="{{ site.baseurl }}/assets/img/posts/2021-08-08-robomimic/lesson_3.png" class="postimagefourfifth"/>
<figcaption>
The mismatch between train and evaluation objective causes problems for policy selection - unlike supervised learning, the best validation loss does not correspond to the best performing policy. We found that the best validation policy is 50 to 100% worse than the best performing policy. Thus, each policy checkpoint needs to be tried directly on the robot – this can be costly.
</figcaption>
{% endfigure %}

### Lesson 4: Observation space and hyperparameters play a large role in policy performance.

{% figure %}
<img src="{{ site.baseurl }}/assets/img/posts/2021-08-08-robomimic/lesson_4.png" class="postimagefourfifth"/>
<figcaption>
We found that observation space choice and hyperparameter selection is crucial for good performance. As an example, not including wrist camera observations can reduce performance by 10 to 45 percent
</figcaption>
{% endfigure %}

### Lesson 5: Using human data for manipulation is promising.

{% figure %}
<img src="{{ site.baseurl }}/assets/img/posts/2021-08-08-robomimic/lesson_5.png" class="postimagefourfifth"/>
<figcaption>
Studying how dataset size impacts performance made us realize that using human data holds much promise. For each task, the bar chart shows how performance changes going from 20% to 50% to 100% of the data. Simpler tasks like Lift and Can require just a fraction of our collected datasets to learn, while more complex tasks like Square and Transport benefit substantially from adding more human data, <b>suggesting that more complex tasks could be addressed by using large human datasets</b>.
</figcaption>
{% endfigure %}

### Lesson 6: Study results transfer to real world.

We collected 200 demonstrations per task, and trained a BC-RNN policy <b>using identical hyperparameters to simulation, with no hyperparameter tuning</b>. We see that in most cases, performance and insights on what works in simulation transfer well to the real world.

{% figure %}

<figure class="postfigurethird">
  <video autoplay loop muted playsinline controls class="postimage_unpadded">
    <source src="{{ site.baseurl }}/assets/img/posts/2021-08-08-robomimic/raw_rollout_lift_eval_bright.mp4" type="video/mp4">
  </video>
  <figcaption>
  <b>Lift (Real).</b> 96.7% success rate. Nearly matches performance in simulation (100%).
  </figcaption>
</figure>

<figure class="postfigurethird">
  <video autoplay loop muted playsinline controls class="postimage_unpadded">
    <source src="{{ site.baseurl }}/assets/img/posts/2021-08-08-robomimic/can_eval_success_bright.mp4" type="video/mp4">
  </video>
  <figcaption>
  <b>Can (Real).</b> 73.3% success rate. Nearly matches performance in simulation (100%).
  </figcaption>
</figure>

<figure class="postfigurethird">
  <video autoplay loop muted playsinline controls class="postimage_unpadded">
    <source src="{{ site.baseurl }}/assets/img/posts/2021-08-08-robomimic/tool_hang_real_succ_bright.mp4" type="video/mp4">
  </video>
  <figcaption>
  <b>Tool Hang (Real).</b> 3.3% success rate. Far from simulation (67.3%) - the real task is harder.
  </figcaption>
</figure>

{% endfigure %}


Below, we present examples of policy failures on the Tool Hang task, which illustrate its difficulty, and the large room for improvement.


{% figure %}

<figure class="postfigurequarter">
  <video autoplay loop muted playsinline controls class="postimage_unpadded">
    <source src="{{ site.baseurl }}/assets/img/posts/2021-08-08-robomimic/tool_hang_insert_miss_5x_bright.mp4" type="video/mp4">
  </video>
  <figcaption>
  Insertion Miss
  </figcaption>
</figure>

<figure class="postfigurequarter">
  <video autoplay loop muted playsinline controls class="postimage_unpadded">
    <source src="{{ site.baseurl }}/assets/img/posts/2021-08-08-robomimic/tool_hang_fail_insert_5x_bright.mp4" type="video/mp4">
  </video>
  <figcaption>
  Failed Insertion
  </figcaption>
</figure>

<figure class="postfigurequarter">
  <video autoplay loop muted playsinline controls class="postimage_unpadded">
    <source src="{{ site.baseurl }}/assets/img/posts/2021-08-08-robomimic/tool_hang_fail_tool_grasp_2_5x_bright.mp4" type="video/mp4">
  </video>
  <figcaption>
  Failed Tool Grasp
  </figcaption>
</figure>

<figure class="postfigurequarter">
  <video autoplay loop muted playsinline controls class="postimage_unpadded">
    <source src="{{ site.baseurl }}/assets/img/posts/2021-08-08-robomimic/tool_hang_tool_drop_5x_bright.mp4" type="video/mp4">
  </video>
  <figcaption>
  Tool Drop
  </figcaption>
</figure>

<figcaption>
Failures which illustrate the difficulty of the Tool Hang task.
</figcaption>

{% endfigure %}

We also show that results from our observation space study hold true in the real world -- visuomotor policies benefit strongly from wrist observations and pixel shift randomization.


{% figure %}

<figure class="postfigurethird">
  <video autoplay loop muted playsinline controls class="postimage_unpadded">
    <source src="{{ site.baseurl }}/assets/img/posts/2021-08-08-robomimic/can_no_wrist_5x_bright.mp4" type="video/mp4">
  </video>
  <figcaption>
  <b>Can (no Wrist).</b> 43.3% success rate (compared to 73.3% with wrist).
  </figcaption>
</figure>

<figure class="postfigurethird">
  <video autoplay loop muted playsinline controls class="postimage_unpadded">
    <source src="{{ site.baseurl }}/assets/img/posts/2021-08-08-robomimic/can_no_rand_5x_bright.mp4" type="video/mp4">
  </video>
  <figcaption>
  <b>Can (no Rand).</b> 26.7% success rate (compared to 73.3% with randomization).
  </figcaption>
</figure>

<figcaption>
Without wrist observations (left) the success rate decreases from 73.3% to 43.3%. Without pixel shift randomization (right), the success rate decreases from 73.3% to 26.7%.
</figcaption>
{% endfigure %}


## Takeaways

{% figure %}
<img src="{{ site.baseurl }}/assets/img/posts/2021-08-08-robomimic/final_task_8.jpeg" class="postimagethird"/>
{% endfigure %}

1. Learning from large multi-human datasets can be challenging.<br><br>
2. Large multi-human datasets hold promise for endowing robots with dexterous manipulation capabilities.<br><br>
3. Studying this setting in simulation can enable reproducible evaluation and insights can transfer to real world.


<hr>

Please see the [robomimic website](https://arise-initiative.github.io/robomimic-web/) for more information. 

This blog post is based on the following paper:

- ["What Matters in Learning from Offline Human Demonstrations for Robot Manipulation"](https://arxiv.org/abs/2108.03298) by Ajay Mandlekar, Danfei Xu, Josiah Wong, Soroush Nasiriany, Chen Wang, Rohun Kulkarni, Li Fei-Fei, Silvio Savarese, Yuke Zhu, and Roberto Martín-Martín.

