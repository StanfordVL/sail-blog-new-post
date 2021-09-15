---
layout: post
title: "Our Journey towards Data-Centric AI: A Retrospective"
short-summary: "A retrospective narrative from the Hazy research lab on our work in data-centric AI, and current efforts on engaging the broader machine learning community."
summary: "A retrospective narrative from the Hazy research lab on our work in data-centric AI, and current efforts on engaging the broader machine learning community."
feature-img: "/assets/img/posts/2021-09-15-data-centric-ai-retrospective/robot.png"
thumbnail: "/assets/img/posts/2021-09-15-data-centric-ai-retrospective/robot.png"
author: <a href="https://cs.stanford.edu/people/chrismre">Christopher Ré</a> and <a href="https://krandiash.github.io">Karan Goel</a>
tags: [ml, machine learning, data centric ai, retrospective]
draft: True
---
{% figure %}
<img class="postfigurehalf" src="{{ site.baseurl }}/assets/img/posts/2021-09-15-data-centric-ai-retrospective/header.png"/>
{% endfigure %}

This article provides a brief, _biased_ retrospective of our road to data-centric AI. Our hope is to provide an entry point for people interested in this area, which has been scattered to the nooks and crannies of AI—even as it drives some of our favorite products, advancements, and benchmark improvements.

We’re collecting pointers to these resources on [GitHub](https://github.com/hazyresearch/data-centric-ai), and plan to write a few more articles about exciting new directions. We hope to engage with folks who are excited about data-centric AI in an upcoming HAI workshop in November — folks like you!

[![Generic badge](https://img.shields.io/badge/GitHub-Data--Centric%20AI%20Resource-informational)](https://github.com/hazyresearch/data-centric-ai)
[![Generic badge](https://img.shields.io/badge/Mailing%20List-Data--Centric%20AI-green)](https://groups.google.com/forum/#!forum/data-centric-ai/join)

---------

Starting in about 2016, researchers from our lab — [the Hazy Research lab](https://hazyresearch.stanford.edu/) — circled through academia and industry [giving talks](https://www.youtube.com/watch?v=iSQHelJ1xxU) about an intentionally provocative idea: machine learning (ML) models—long the darlings of researchers and practitioners—were no longer the center of AI. In fact, models were becoming commodities. Instead, we claimed that it was the training data that would drive progress towards more performant ML models and systems.

To underscore this, we had taglines like “AI is driven by data—not code” or worse ”[Training data is the _new_ new oil](https://www.youtube.com/watch?v=iSQHelJ1xxU)”. We started building systems [championed by little octopuses wearing snorkels](http://snorkel.ai). Eventually, we turned to others and called this “Software 2.0” (inspired by [Karpathy’s post](https://karpathy.medium.com/software-2-0-a64152b37c35). Others have since termed it data-centric AI, and recently Andrew Ng gave [a great talk](https://www.youtube.com/watch?v=06-AZXmwHjo) about his perspective on this direction.

Our view that models were becoming a commodity was heretical for a few reasons.

First, people often think of data as a static thing. After all, data literally means “that which is given”. For most ML people, they download an off-the-shelf dataset, drop it into a PyTorch dataloader, and plug-and-play: losses go down, accuracy goes up, and the data is a mere accessory.

But to an engineer in the wild, the training data is never “that which is given”. It is the result of a process — usually a dirty, messy process that is critical and underappreciated.

{% figure %}
<img class="postimage" src="{{ site.baseurl }}/assets/img/posts/2021-09-15-data-centric-ai-retrospective/doggo.png"/>
<figcaption>
An engineer and their training data in the wild. <it>Credit: Vickie Shelton.</it>
</figcaption>
{% endfigure %}


Still, we had hope. In applications, we took time to clean and merge data. We engineered it. We began to talk about how AI and ML systems were driven by this data, how they were programmed by this data. This led to understandably (obtuse) names like “data programming”.

Unfortunately, we were telling people to put on galoshes, jump into the sewer that is your data, and splash around. Not an easy sales pitch for researchers used to life in beautiful PyTorch land.

We started to recognize that [model-itis is a real problem](https://arxiv.org/pdf/1909.05372.pdf). With some friends at Apple, we realized that teams would often spend time writing new models instead of understanding their problem—and its expression in data—more deeply. We weren’t the only ones thinking this way, lots of no-code AI folks like [Ludwig](https://ludwig-ai.github.io), [H2O](https://www.h2o.ai/), [DataRobot](https://www.datarobot.com) were too. We began to argue that this aversion to data didn’t really lead to a great use of time. To make matters worse, 2016-2017 was a thrilling time to be in ML. Each week a new model came out, and each week, it felt like we were producing demos that we couldn’t dream of a decade earlier.

Despite this excitement, it was clear to us that success or failure to a level usable in applications we cared about—in medicine, at large technology companies or even pushing the limits on benchmarks—wasn’t really tied to models per se. That is, the advances were impressive, but they were hitting diminishing returns. You can see this in benchmarks, where most of the progress after 2017 is fueled by new advances in augmentations, weak supervision, and other issues of how you feed machines data. In round numbers, ten points of accuracy were due to those—while (by and large) model improvements were squeaking out a few tenths in accuracy points.

At the time, many of the folks who are now converts have shared with us that they were skeptical of our view of the future. We get it, our stupid jokes and general demeanor didn’t inspire confidence. But we weren’t totally insane. This idea has become mainstream and widespread. Our friends at Google in Ads, Gmail, YouTube and Apple extended to us a level of technical trust that we hope we’ve repaid. You’ve probably used some of the products that have incorporated these crazy ideas in the last few minutes. [The Octopus](https://snorkel.ai) is now widely used in the enterprise, and we’re just at the beginning!

This blog post is an incomplete, biased retrospective of this road. We’ll close with two thoughts:

1. There is a data-centric research agenda inside AI. It’s intellectually deep, and it has been lurking at the core of AI progress for a while. Perhaps by calling it out we can make even more progress on an important viewpoint. 
2. We’d love to provide entry points for folks interested in this area. Our results are scattered in a number of different research papers, and we’d enjoy writing a survey (if anyone is interested -- we have a [form](https://docs.google.com/forms/d/e/1FAIpQLSf5UcTJnvMIcLzxvTgac5Jdvyry3u2XsewMrXFosgKtWTTGxA/viewform?usp=sf_link)!). We’ve opted to be biased about what influenced us the most to try to present a coherent story here. Necessarily, this means we’re leaving out amazing work. Apologies, please send us notes and corrections.

On our end, we’ll do our best to build this [data-centric community up on GitHub](https://github.com/hazyresearch/data-centric-ai), with a collage of exciting related papers and lines of work. If you’re new to the area, use it as a pedagogical resource, and if you’re a veteran, please go ahead and send us PRs and contributions so we can expand the discussion! We’re gathering [real-world case studies](https://github.com/HazyResearch/data-centric-ai/tree/main/case-studies), so if you work on real applications that have benefited from a data-centric viewpoint (in academia, industry or anywhere), please don’t hesitate to reach out at [kgoel@cs.stanford.edu](mailto:kgoel@cs.stanford.edu) or create an Issue on the Github so we can bring your experiences into the fold. 

_A more informal version of this blog can be found [here](https://hazyresearch.stanford.edu/data-centric-ai)._