---
layout: post
title: "How to improve user experience (and behavior): Three papers from Stanford's Alexa Prize team"
short-summary: "Strategies for understanding user dissatisfaction, handling offensiveness, and increasing user initiative, to improve conversational experience."
summary: "Strategies for understanding user dissatisfaction, handling offensiveness, and increasing user initiative, to improve conversational experience."
feature-img: "/assets/img/posts/2022-01-24-alexa-sigdial/images/chirpy_logo_optimized.svg"
thumbnail: "/assets/img/posts/2022-01-24-alexa-sigdial/images/chirpy_logo_optimized.svg"
author: <a href="https://www.linkedin.com/in/ameliahardy">Amelia Hardy</a>, <a href="https://www.linkedin.com/in/haojun-li">Haojun Li</a>, and <a href="https://cs.stanford.edu/people/abisee/">Abigail See</a>
tags: [ml, machine learning, nlp]
draft: True
---
## Introduction

In 2019, Stanford entered the [Alexa Prize Socialbot Grand Challenge 3](https://www.google.com/url?q=https://developer.amazon.com/alexaprize/challenges/past-challenges/challenge3&sa=D&source=editors&ust=1643077986229779&usg=AOvVaw0mZqjEuNWtTeW_HL6vpEwg) for the first time, with its bot[ Chirpy Cardinal](https://www.google.com/url?q=https://stanfordnlp.github.io/chirpycardinal/&sa=D&source=editors&ust=1643077986230129&usg=AOvVaw0PSQLr7ujAiUXzz9MflQE2), which went on to win 2nd place in the competition. In [our previous post](https://www.google.com/url?q=http://ai.stanford.edu/blog/chirpy-cardinal/&sa=D&source=editors&ust=1643077986230491&usg=AOvVaw15s3Z53omx8Z2--EFbFWnM), we discussed the technical structure of our socialbot and how developers can use our [open-source code](https://www.google.com/url?q=https://github.com/stanfordnlp/chirpycardinal&sa=D&source=editors&ust=1643077986230799&usg=AOvVaw23mdMqhgTjpUsPlAXvrkwd) to develop their own. In this post we share further research conducted while developing Chirpy Cardinal to discover common pain points that users encounter when interacting with socialbots, and strategies for addressing them.

The Alexa Prize is a unique research setting, as it allows researchers to study how users interact with a bot when doing so solely for their own motivations. During the competition, US-based Alexa users can say the phrase "let's chat" to speak in English to an anonymous and randomly-selected competing bot. They are free to end the conversation at any time. Since Alexa Prize socialbots are intended to create as natural an experience as possible, they should be capable of long, open-domain social conversations with high coverage of topics. We observed that Chirpy users were interested in many different subjects, from current events (e.g., the coronavirus) to pop culture (e.g., the movie *Frozen 2*) to personal interests (e.g,. their pets). Chirpy achieves its coverage of these diverse topics by using a modular design that combines both neural generation and scripted dialogue, as described in our [previous post](https://www.google.com/url?q=http://ai.stanford.edu/blog/chirpy-cardinal/&sa=D&source=editors&ust=1643077986232156&usg=AOvVaw3M4sXPACj4_01KajypLDfN).


We used this setting to study three questions about socialbot conversations:

1. [What do users complain about, and how can we learn from the complaints to improve neurally generated dialogue?](#1-understanding-and-predicting-user-dissatisfaction)
2. [What strategies are effective and ineffective in handling and deterring offensive user behavior?](#2-handling-offensive-users)
3. [How can we shift the balance of power, such that both users and the bot are meaningfully controlling the conversation?](#3-increasing-user-initiative)

We've published papers on each of these topics at [SIGDIAL 2021](https://www.google.com/url?q=https://www.sigdial.org/files/workshops/conference22/&sa=D&source=editors&ust=1643077986234101&usg=AOvVaw3g1NgXhcztFbTrWElRxCmd) and in this post, we'll share key findings which provide practical insights for both chatbot researchers and developers.

## 1. Understanding and Predicting User Dissatisfaction

[paper](https://www.google.com/url?q=https://sigdial.org/sites/default/files/workshops/conference22/Proceedings/pdf/2021.sigdial-1.1.pdf&sa=D&source=editors&ust=1643077986235278&usg=AOvVaw0pEUVzPwaFJ1MfMpLOQA9o)|[video](https://www.google.com/url?q=https://drive.google.com/file/d/1MLBT54DTM2qwXoOi-ZYR0z5TrbPnL_oz/view?usp%3Dsharing&sa=D&source=editors&ust=1643077986235665&usg=AOvVaw0BOHA6t9ytorNh1vgv12pB)

Neural generative dialogue models like DialoGPT[^dialogpt], Meena[^meena], and BlenderBot[^blender] use large pretrained neural language models to generate responses given a dialogue history. These models perform well when evaluated by crowdworkers in carefully-controlled settings--typically written conversations with certain topical or length constraints.

However, real-life settings like the Alexa Prize are not so tidy. Users have widely varying expectations and personalities, and require fast response times as they speak with the bot in home environments that might feature cross-talk and background noise. Through Chirpy Cardinal, we have a unique opportunity to investigate how modern neural generative dialogue models hold up in this kind of environment.

Chirpy Cardinal uses a GPT2-medium model fine-tuned on the EmpatheticDialogues[^empatheticdialogues] dataset to hold short discussions with users about their everyday experiences and emotions. Particularly during the pandemic, we found it was important for Chirpy to ask users about these issues. Though larger and more powerful pretrained generative models are available, we used GPT2-medium due to budget and latency constraints.

{% figure %}
<img class="postimage" src="{{ site.baseurl }}/assets/img/posts/2022-01-24-alexa-sigdial/images/image8.png"/>
{% endfigure %}

While the GPT2-medium model is capable of chatting about these simple topics for a few utterances, discussions that extend longer tend to derail. Sooner or later, the bot gives a response that doesn\'t quite make sense, and it\'s hard for the user or the model to recover the conversation.

To understand how these conversations are derailing, we defined 7 types of errors made by the neural generative model -- repetition, redundant questions, unclear utterances, hallucination, ignoring, logical errors, and insulting utterances. After annotating a sample of user conversations, we found that bot errors were common, with over half (53%) of neural-generated utterances containing some kind of error.

We also found that due to the challenging noisy environment (which may involve background noise, cross-talk, and ASR errors), almost a quarter (22%) of user utterances were incomprehensible, even to a human annotator. This accounts for some of the more basic bot errors, such as ignoring, hallucination, unclear and repetitive utterances.

Of the remaining bot errors, redundant questions and logical errors are particularly common, indicating that better reasoning and use of the conversational history are a priority for neural generative model development.

We also tracked 9 ways that users express dissatisfaction, such as asking for clarification, criticising the bot, and ending the conversation. Though there is a relationship between bot errors and user dissatisfaction, the correlation is noisy. Even after a bot error, many users do not express dissatisfaction, instead attempting to continue the conversation. This is particularly true after logical errors, in which the bot shows a lack of real-world knowledge or commonsense -- some kind-hearted users even take this as an opportunity to educate the bot. Conversely, some users express dissatisfaction unrelated to any obvious bot error -- for example, users have widely differing expectations regarding what kinds of personal questions are appropriate from the bot.

Having better understood how and why users express dissatisfaction, we asked: can we learn to predict dissatisfaction, and thus prevent it before it happens?

{% figure %}
<img class="postimage" src="{{ site.baseurl }}/assets/img/posts/2022-01-24-alexa-sigdial/images/image1.png"/>
{% endfigure %}

With the user conversations collected during the competition, we trained a model to predict the probability that a certain bot utterance would lead the user to express dissatisfaction. Given the noisy correlation between bot errors and user dissatisfaction, this is inherently challenging. Despite this noise, our predictor model was able to find signal in the users' dissatisfaction.

Once trained, our dissatisfaction predictor can be used mid-conversation to choose between multiple alternative neural-generated bot utterances. Through human evaluation, we found that the bot responses chosen by the predictor -- i.e., those judged least likely to cause user dissatisfaction -- are overall better quality than randomly chosen responses.

Though we have not yet incorporated this feedback loop into Chirpy Cardinal, our method demonstrates one viable way to implement a semi-supervised online learning method to continuously improve a neural generative dialogue system.

## 2. Handling Offensive Users

[paper](https://www.google.com/url?q=https://sigdial.org/sites/default/files/workshops/conference22/Proceedings/pdf/2021.sigdial-1.58.pdf&sa=D&source=editors&ust=1643077986242432&usg=AOvVaw2WvGYbnpNB1xyU67nzA98N) | [video](https://www.google.com/url?q=https://drive.google.com/file/d/12ePMS49YoNtFgy_uoQhP2DeL7w85PvL_/view?usp%3Dsharing&sa=D&source=editors&ust=1643077986242830&usg=AOvVaw0RSX061Yry32ifSYx0cMTx)

Voice assistants are becoming increasingly popular, and with their popularity, they are subject to growing abuse from their user populations. We estimate that more than 10% of user conversations with our bot, Chirpy Cardinal, contain profanity and overtly offensive language. While there is a large body of prior work attempting to address this issue, most prior approaches use qualitative metrics based on surveys conducted in lab settings. In this work, we conduct a large-scale quantitative evaluation of response strategies against offensive users in-the-wild. In our experiments, we found that politely rejecting the user's offense while redirecting the user to an alternative topic is the best strategy in curbing offenses.

Informed by prior work, we test the following 4 hypotheses:

1. **Redirect** - Inspired by Brahnam[^brahnam05], we hypothesize that using explicit redirection when responding to an offensive user utterance is an effective strategy. For example, "I'd rather not talk about that. So, who's your favorite musician?"
2. **Name** - Inspired by Suler[^suler04] and Chen and Williams[^chenwilliams20], we hypothesize that including the user's name in the bot's response is an effective strategy. For example, "I'd rather not talk about that, Peter."
3. **Why** - Inspired by Shapiro et al.[^shapiro14], we hypothesize that politely asking the user the reason why they made an offensive remark invites them to reflect on their behavior, reducing future offenses. For example, "Why would you say that?"
4. **Empathetic & Counter** - Inspired by Chin et al.[^chin20], we hypothesize that empathetic responses are more effective than generic avoidance responses, while counter-attack responses make no difference. For example, an empathetic response would be "If I could talk about it I would, but I really can't. Sorry to disappoint", and a counter-attack response would be "That's a very suggestive thing to say. I don't think we should be talking about that."

We constructed the responses crossing multiple factors listed above. For example, avoidance + name + redirect would yield the utterance "I'd rather not talk about that (*avoidance*), Peter (*name*). So, who's your favorite musician? (*redirect*)"

To measure the effectiveness of a response strategy, we propose 3 metrics:

1. **Re-offense** - measured as the number of conversations that contained another offensive utterance after the initial bot response.
2. **End** - measured as the length of the conversation after bot response assuming no future offenses.
3. **Next** - measured as the number of turns passed until the user offends again.

We believe that these metrics measure the effectiveness of a response strategy more directly than user ratings as done in Cohn et al.[^cohn19] which measure the overall quality of the conversation.

{% figure %}
<img class="postimage_75" src="{{ site.baseurl }}/assets/img/posts/2022-01-24-alexa-sigdial/images/image9.png"/>
{% endfigure %}

The figure above shows the differences of strategies on the Re-offense ratio. As we can see, strategies with (*redirects*) performed significantly better than strategies without redirects, reducing re-offense rate by as much as 53%. Our pairwise hypothesis tests further shows that using user's name with a redirect further reduces re-offense rate by about 6%, and that asking the user why they made an offensive remark had a 3% **increase** in re-offense rate which shows that asking the user why only invites user re-offenses instead of self-reflection. Empathetic responses also reduced re-offense rate by 3%, while counter responses did not have any significant effect.

{% figure %}
<img class="postimagehalf" src="{{ site.baseurl }}/assets/img/posts/2022-01-24-alexa-sigdial/images/image7.png"/>
<img class="postimagehalf" src="{{ site.baseurl }}/assets/img/posts/2022-01-24-alexa-sigdial/images/image3.png"/>
{% endfigure %}

The figure on the left shows the differences in average number of turns until the next re-offense (*Next*), and the figure on the right shows the differences in average number of turns until the end of the conversation (*End*). We again see that strategies with (*redirects* are able to significantly prolong a non-offensive conversation. This further shows that redirection is incredibly effective method to curb user offenses.

The main takeaway from this is that **the bot should always empathetically respond to user offenses with a redirection, and use the user\'s name whenever possible.**

Despite the empirical effectiveness of the passive avoidance and redirection strategy, we would like to remind researchers of the societal dangers of adopting similar strategies. Since most voice-based agents have a default female voice, these strategies could further gender stereotypes and set unreasonable expectations of how women would react to verbal abuse in the real world [^curryreiser19] [^west19] [^curry20]. Thus, caution must be taken when deploying these strategies.

## 3. Increasing User Initiative

[paper](https://www.google.com/url?q=https://sigdial.org/sites/default/files/workshops/conference22/Proceedings/pdf/2021.sigdial-1.11.pdf&sa=D&source=editors&ust=1643077986251387&usg=AOvVaw3V-pqs-4qBUJdc7_rTOVOn)|[video](https://www.google.com/url?q=https://drive.google.com/file/d/1jZPThbl6Y7uHGP0HKX8n3Uroflkb_74O/view?usp%3Dsharing&sa=D&source=editors&ust=1643077986251761&usg=AOvVaw10jXhy2kar7zP2f7vskHDN)

Conversations are either controlled by the user (for example, bots such as Apple's Siri, which passively waits for user commands) or the bot (for example, CVS's customer service bot, which repeatedly prompts the user for specific pieces of information).

This property - which agent has control at a given moment - is called initiative.

It wouldn't be fun to go to a cocktail party and have a single person choose every topic, never giving you the opportunity to share your own interests. It's also tedious to talk to someone who forces you to carry the conversation by refusing to bring up their own subjects. Ideally, everyone would take turns responding to prompts, sharing information about themselves, and introducing new topics. We call this pattern of dialogue **mixed initiative** and hypothesize that just as it's an enjoyable type of human-human social conversation, it's also a more engaging and desirable form of human-bot dialogue.

We designed our bot, Chirpy Cardinal, to keep conversations moving forward by asking questions on every turn. Although this helped prevent conversations from stagnating, it also made it difficult for users to take initiative. In our data, we observe users complaining about this, with comments such as *you ask too many questions*, or *that's not what I wanted to talk about*.

Since our goal in studying initiative was to make human-bot conversations more like human-human ones, we looked to research on human dialogue for inspiration.

Based on this research, we formed three hypotheses for how to increase user initiative.

The images below show the types of utterances we experimented with as well as representative user utterances. Per Alexa Prize competition rules, these are not actual user utterances received by our bot.

{% figure %}
<img class="postimage_75" src="{{ site.baseurl }}/assets/img/posts/2022-01-24-alexa-sigdial/images/image6.png"/>
{% endfigure %}

#### 1. Giving statements instead of questions

In human dialogue research [^whittakerwalker90], the person asking a question has initiative, since they are giving a direction that the person answering follows. By contrast, an open-ended statement gives the listener an opportunity to take initiative. This was the basis of our first strategy: **using statements instead of questions**.

{% figure %}
<img class="postimage_75" src="{{ site.baseurl }}/assets/img/posts/2022-01-24-alexa-sigdial/images/image2.png"/>
{% endfigure %}

#### 2. Sharing personal information

Work on both human-human [^collinsmiller94] and human-bot [^lee20] dialogue has found that personal self disclosure has a reciprocal effect. If one participant shares about themself, then the other person is more likely to do the same. We hypothesized that **if Chirpy gave personal statements rather than general ones, then users would take initiative and reciprocate**.

{% figure %}
<img class="postimagehalf" src="{{ site.baseurl }}/assets/img/posts/2022-01-24-alexa-sigdial/images/image4.png"/> 
<img class="postimagehalf" src="{{ site.baseurl }}/assets/img/posts/2022-01-24-alexa-sigdial/images/image5.png"/>
{% endfigure %}

The figure on the left is an example of a conversation with back-channeling, the right, without. In this case, back-channeling allows the user to direct the conversation towards what they want (getting suggestions) rather than forcing them to talk about something they're not interested in (hobbies).

#### 3. Introducing back-channeling

Back-channels, such as "hmm", "I see", and "mm-hmm", are brief utterances which are used as a signal from the listener to the speaker that the speaker should continue taking initiative. Our final hypothesis was that they could be used in human-bot conversation to the same effect, i.e. that **if our bot back-channeled, then the user would direct the conversation**.

#### Experiments and results
To test these strategies, we altered different components of our bot. We conducted small experiments, only altering a single turn of conversation, to test questions vs statements and personal vs general statements. To test the effect of replacing statements with questions on a larger number of turns, we altered components of our bot that used neurally generated dialogue, since these were more flexible to changing user inputs. Finally, we experimented with back-channeling in a fully neural module of our bot.


Using a set of automated metrics, which we validated using manual annotations, we found the following results, which provide direction for future conversational design:

1. Using statements alone outperformed questions or combined statements and questions 
2. Giving personal opinion statements (e.g. "I like Bojack Horseman") was more effective than both personal experience statements (e.g. "I watched Bojack Horseman yesterday") and general statements (e.g. "Bojack Horseman was created by Raphael Bob-Waksberg and Lisa Hanawalt")
3. As the number of questions decreased, user initiative increased
4. User initiative was greatest when we back-channeled 33% of the time (as opposed to 0%, 66%, or 100%)

Since these experiments were conducted in a limited environment, we do not expect that they would transfer perfectly to all social bots; however, we believe that these simple yet effective strategies are a promising direction for building more natural conversational AI.

## 4. Listen with empathy

Each of our projects began with dissatisfied users who told us, in their own words, what our bot could do better. By conducting a systematic analysis of these complaints, we gained a more precise understanding of what specifically was bothering users about our neurally generated responses. Using this feedback, we trained a model which was able to successfully predict when a generated response might lead the conversation astray. At times, it was the users who would make an offensive statement. We studied these cases and determined that an empathetic redirection, which incorporated the users name, was most effective at keeping the conversation on track. Finally, we experimented with simply saying less and creating greater opportunities for the user to lead the conversation. When presented with that chance, many took it, leading to longer and more informative dialogues. 

Across all of our work, the intuitive principles of human conversation apply to socialbots: be a good listener, respond with empathy, and when you're given feedback and the opportunity to learn, take it.

------------------------------------------------------------------------

[^dialogpt]: Zhang, Yizhe, Siqi Sun, Michel Galley, Yen-Chun Chen, Chris Brockett, Xiang Gao, Jianfeng Gao, Jingjing Liu, and Bill Dolan. Dialogpt: Large-scale generative pre-training for conversational response generation](https://www.google.com/url?q=https://arxiv.org/abs/1911.00536&sa=D&source=editors&ust=1643077986262380&usg=AOvVaw1khQv7HglJrP1gK8dkiE3n).\" arXiv preprint arXiv:1911.00536 (2019).

[^meena]: Adiwardana, Daniel, Minh-Thang Luong, David R. So, Jamie Hall, Noah Fiedel, Romal Thoppilan, Zi Yang et al. [Towards a human-like open-domain chatbot](https://www.google.com/url?q=https://arxiv.org/abs/2001.09977&sa=D&source=editors&ust=1643077986262944&usg=AOvVaw3Pbae_MvzxjvmdBhHJ9KzL) arXiv preprint arXiv:2001.09977 (2020).

[^blender]: Roller, Stephen, Emily Dinan, Naman Goyal, Da Ju, Mary Williamson, Yinhan Liu, Jing Xu et al. [Recipes for building an open-domain chatbot](https://www.google.com/url?q=https://arxiv.org/abs/2004.13637&sa=D&source=editors&ust=1643077986263477&usg=AOvVaw2YmyyrWz7jQOkz8JkXDwjz) arXiv preprint arXiv:2004.13637 (2020).

[^empatheticdialogues]: Hannah Raskin, Eric Michael Smith, Margaret Li, and Y-Lan Boureau. 2019. [Towards empathetic open-domain conversation models: A new benchmark and dataset.](https://www.google.com/url?q=https://arxiv.org/pdf/1811.00207.pdf&sa=D&source=editors&ust=1643077986264028&usg=AOvVaw32mtRQhV_DhtjxHkvAS8Jw) In Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics, pages 5370-5381, Florence, Italy. Association for Computational Linguistics.

[^brahnam05]: Sheryl Brahnam. 2005. Strategies for handling cus- tomer abuse of ECAs. In *Proc. Interact 2005 work- shop Abuse: The darker side of Human-Computer Interaction*, pages 62–67.

[^suler04]: John Suler. 2004. The online disinhibition effect. *Cyberpsychology & behavior*, 7(3):321–326.

[^chenwilliams20]: Xiangyu Chen and Andrew Williams. 2020. [Improving Engagement by Letting Social Robots Learn and Call Your Name](https://doi.org/10.1145/3371382.3378355). In *Companion of the 2020 ACM/IEEE International Conference on Human-Robot Interaction*, HRI ’20, page 160–162, New York, NY, USA. Association for Computing Machinery.

[^shapiro14]: Shauna Shapiro, Kristen Lyons, Richard Miller, Britta Butler, Cassandra Vieten, and Philip Zelazo. 2014. [Contemplation in the Classroom: a New Direction for Improving Childhood Education](https://link.springer.com/article/10.1007%2Fs10648-014-9265-3). *Educational Psychology Review*, 27.

[^chin20]: Hyojin Chin, Lebogang Wame Molefi, and Mun Yong Yi. 2020. Empathy Is All You Need: How a Conversational Agent Should Sespond to Verbal Abuse. In *Proceedings of the 2020 CHI Conference on Human Factors in Computing Systems*, pages 1–13.

[^cohn19]: Michelle Cohn, Chun-Yen Chen, and Zhou Yu. 2019. [A large-scale user study of an Alexa Prize chatbot: Effect of TTS dynamism on perceived quality of social dialog](https://aclanthology.org/W19-5935/). In *Proceedings of the 20th Annual SIGdial Meeting on Discourse and Dialogue*, pages 293– 306, Stockholm, Sweden. Association for Computational Linguistics.

[^curryreiser19]: Amanda Cercas Curry and Verena Rieser. 2019. [A crowd-based evaluation of abuse response strategies in conversational agents](https://aclanthology.org/W19-5942/). In *Proceedings of the 20th Annual SIGdial Meeting on Discourse and Dialogue*, pages 361–366, Stockholm, Sweden. Association for Computational Linguistics.

[^west19]: Mark West, Rebecca Kraut, and Han Ei Chew. 2019. I’d blush if i could: closing gender divides in digital skills through education.

[^curry20]: Amanda Cercas Curry, Judy Robertson, and Verena Rieser. 2020. [Conversational assistants and gender stereotypes: Public perceptions and desiderata for voice personas](https://aclanthology.org/2020.gebnlp-1.7/). In *Proceedings of the Second Work- shop on Gender Bias in Natural Language Processing*, pages 72–78, Barcelona, Spain (Online). Association for Computational Linguistics.

[^whittakerwalker90]: Marilyn Walker and Steve Whittaker. 1990. [Mixed initiative in dialogue: An investigation into discourse segmentation](https://dl.acm.org/doi/10.3115/981823.981833).    In *Proceedings of the 28th Annual Meeting on Association for Computational Linguistics*,  ACL  ’90, page 70–78, USA. Association for Computational Linguistics.

[^collinsmiller94]: Nancy Collins and Lynn Miller. 1994. [Self-disclosure and liking: A meta-analytic review](https://doi.apa.org/doiLanding?doi=10.1037%2F0033-2909.116.3.457). *Psychological bulletin*, 116:457–75.

[^lee20]: Yi-Chieh Lee, Naomi Yamashita, Yun Huang, and Wai Fu. 2020. “I hear you, I feel you”: Encouraging deep self-disclosure through a chatbot. In *Proceedings of the 2020 CHI Conference on Human Factors in Computing Systems*, CHI ’20, page 1–12, New York, NY, USA. Association for Computing Machinery.
