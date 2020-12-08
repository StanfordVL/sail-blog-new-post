# [SAIL Blog](http://ai.stanford.edu/blog/)
## Why write for SAIL blog
It's a great way to let a broader set of people get exposed to your work and help represent SAIL. So if you have recently published or ongoing research that you feel could benefit from being shared more widely in this format, or if one of your new year's resolution is to work on scientific communication, please reach out to the  email the [editors](http://ai.stanford.edu/blog/about/) and we will let you know the process from there.

## How to write a new blog post
1. Confirm your paper(s) are a good fit to write a blog post for (see requirements [here](https://docs.google.com/document/d/1PWuruWbudrAwTI_iJYC-ZkjAd-hILcisL9Ao_VOUtx4/edit?usp=sharing)). You are encouraged to email the [editors](http://ai.stanford.edu/blog/about/) confirm your paper is a good fit and ask any quetsions. 
2. Get a draft of your post in Google doc, and email the [editors](http://ai.stanford.edu/blog/about/)  to get an editor assigned to the draft. More detailed instructions on that [here](https://docs.google.com/document/d/1PWuruWbudrAwTI_iJYC-ZkjAd-hILcisL9Ao_VOUtx4/edit?usp=sharing). 
3. One or two editors will help you finalize the draft with feedback. Once you have a draft that is finalized, you need to create a pull request with markdown and images of your post. First, fork the repo, clone the fork and pull source branch
4. Follow [these instructions](https://docs.google.com/document/d/1zE6GyVmlAa04mGX9QF9Ip05VVJq4RUnKmlVaUqub9as/edit?usp=sharing) to convert your post from google doc to markdown and submit a pull request. 
5. Once you've submitted a pull request and email your editor to let them know, we'll typically merge it within a few days and publicize via our mailing list and Twitter. Feel free to request specific wording for the tweet promoting it.

### Formatting Tips

For embeddings figures, we generally recommend using {% figure %} wrapping and our supported CSS classes. See examples below: 

Single image that is 3/4th of the width of text:

    {% figure %}
    <img class="postimage_75" src="{{ site.baseurl }}/assets/img/posts/2019-10-28-learning-from-partners/image13.png"/>
    {% endfigure %}

Two images side by side:

    {% figure %}
    <img class="postimagehalf" src="{{ site.baseurl }}/assets/img/posts/2019-10-28-learning-from-partners/image16.png"/>
    <img class="postimagehalf" src="{{ site.baseurl }}/assets/img/posts/2019-10-28-learning-from-partners/image1.png"/>
    {% endfigure %}

Three videos side by side, with a caption:

    {% figure %}
    <video autoplay loop muted playsinline class="postimagethird">
      <source src="{{ site.baseurl }}/assets/img/posts/2019-11-08-roboturk/robonet.mp4" type="video/mp4">
    </video>
    <video autoplay loop muted playsinline class="postimagethird">
      <source src="{{ site.baseurl }}/assets/img/posts/2019-11-08-roboturk/collecting_2.mp4" type="video/mp4">
    </video>
    <video autoplay loop muted playsinline class="postimagethird">
      <source src="{{ site.baseurl }}/assets/img/posts/2019-11-08-roboturk/collecting_3_crop.mp4" type="video/mp4">
    </video>
    <figcaption>
    Prior data collection methodologies include autonomous data collection (left), human supervision with web interfaces (middle), and human teleoperation with motion interfaces (right).
    </figcaption>
    {% endfigure %}
   
Two rows of three images side by side, with a sub-caption for each image and a caption for overall figure:

    {% figure %}
    <figure class="postfigurethird" >
      <img src="{{ site.baseurl }}/assets/img/posts/2019-11-26-robonet/align_tshirt.gif" class="postimage_unpadded"/>
      <figcaption>
      Kuka can align shirts next to the others
      </figcaption>
    </figure>
    <figure class="postfigurethird" >
      <img src="{{ site.baseurl }}/assets/img/posts/2019-11-26-robonet/color_stripe_front.gif" class="postimage_unpadded"/>
      <figcaption>
      Baxter can sweep the table with cloth
      </figcaption>
    </figure>
    <figure class="postfigurethird">
      <img src="{{ site.baseurl }}/assets/img/posts/2019-11-26-robonet/marker_marker.gif" class="postimage_unpadded"/>
      <figcaption>
      Franka can grasp and reposition the markers
      </figcaption>
    </figure>
    <figure class="postfigurethird" >
      <img src="{{ site.baseurl }}/assets/img/posts/2019-11-26-robonet/move_plate.gif" class="postimage_unpadded"/>
      <figcaption>
      Kuka can move the plate to the edge of the table
      </figcaption>
    </figure>
    <figure class="postfigurethird" >
      <img src="{{ site.baseurl }}/assets/img/posts/2019-11-26-robonet/socks_right.gif" class="postimage_unpadded"/>
      <figcaption>
      Baxter can pick up and reposition socks 
      </figcaption>
    </figure>
    <figure class="postfigurethird">
      <img src="{{ site.baseurl }}/assets/img/posts/2019-11-26-robonet/towel_stack.gif" class="postimage_unpadded" />
      <figcaption>
      Franka can stack the towel on the pile
      </figcaption>
    </figure>
    <figcaption>
    Here you can see examples of visual foresight fine-tuned to perform basic control tasks in three entirely different environments. For the experiments, the target robot and environment was subtracted from RoboNet during pre-training. Fine-tuning was accomplished with data collected in one afternoon.
    </figcaption>
    {% endfigure %}


The full list of CSS classes is this:

    .postimage {
      width: 100%;
      height: auto;
      margin: auto;
    }

    .postimage_unpadded {
      width: 100%;
      height: auto;
      margin: auto;
      padding: 0;
    }

    .postimagesmall {
      display:block;
      margin-left:auto;
      margin-right:auto;
      height: auto;
    }

    .postimageactual {
      display:block;
      margin-left:auto;
      margin-right:auto;
      height: auto;
    }

    .postimagesmaller {
      width: 75%;
      display:block;
      margin-left:auto;
      margin-right:auto;
      height: auto;
    }

    .postimage_75 {
      width: 75%;
      display:block;
      height: auto;
      margin-left:auto;
      margin-right:auto;
    }

    .postimage_50 {
      width: 50%;
      display:block;
      height: auto;
      margin-left:auto;
      margin-right:auto;
    }

    .postimage_25 {
      width: 25%;
      display:block;
      height: auto;
      margin-left:auto;
      margin-right:auto;
    }

    .postimagehalf {
      width: 49%;
      height: auto;
      margin-left:auto;
      margin-right:auto;
      padding: 1%;
    }

    .postimagethird {
      width: 30%;
      height: auto;
      margin-left:auto;
      margin-right:auto;
      padding: 1%;
    }

    .postimagequarter {
      width: 24%;
      height: auto;
      margin-left:auto;
      margin-right:auto;
    }
    
    .postfigurehalf {
      width: 49%;
      height: auto;
      margin-left:auto;
      margin-right:auto;
      padding: 1%;
      display: inline-block;
    }

    .postfigurethird {
      width: 30%;
      height: auto;
      margin-left:auto;
      margin-right:auto;
      padding: 1%;
      display: inline-block;
    }

    .postfigurequarter {
      width: 24%;
      height: auto;
      margin-left:auto;
      margin-right:auto;
      display: inline-block;
    }

  
### Citations

We support [bigfoot](http://www.bigfootjs.com/) pop-up citations and recommend using them. In text, use [^<name>] as follows:

    - **Autonomous Data Collection**: Many data collection mechanisms and algorithms such as Self-Supervised Learning[^SSL][^robonet] 

Then at bottom of file:

    [^robonet]: Dasari, S., Ebert, F., Tian, S., Nair, S., Bucher, B., Schmeckpeper, K., ... & Finn, C. (2019). RoboNet: Large-Scale Multi-Robot Learning. arXiv preprint arXiv:1910.11215.
    [^SSL]: Levine, S., Pastor, P., Krizhevsky, A., & Quillen, D. (2016, October). Learning hand-eye coordination for robotic grasping with large-scale data collection. In International Symposium on Experimental Robotics (pp. 173-184). Springer, Cham.

## For editors 

The way things work is that we have a 'source' branch with all the markdown and jekyll files, and the master branch has the compiled HTML. This master branch is cloned to /afs/.cs/group/ai/www/blog/ and is how we update the site's contents. When dealing with Pull Requests for new blog posts, do the following:
1. Merge the pull request
2. Pull source, and locally run bundle exec jekyll serve (with --future if needbe) to visually check all looks good
3. Do a clean build and push to master
4. Schedule the tweet and email via Mailchimp

For step 3, to do this, go to repo root folder source branch, run `gem install octopress` (if you have not yet), then run ./scripts/build_push_to_master, and lastly go to /afs/.cs/group/ai/www/blog/ and run git pull.

build_push_to_master just does the following:
1. bundle exec jekyll clean
2. export JEKYLL_ENV=production
3. bundle exec jekyll build 
4. 'octopress deploy'

You don't technically need it, you can also just copy the \_site contents after build and commit them to master manually, but octopress deploy is a little shortcut that makes this simpler.
