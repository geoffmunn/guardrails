# How to set up a guardrail LLM

This project is to demonstrate how to set up a guardrail project that only allows particular question types to reach an LLM.

## Why is this important?

Let's say you're employed by a government agency or a company who is very excited about this whole 'AI thing', but the security or legal team are very worried about the chatbot going rogue with answers about Uncle Elon or el Presidente Trump (or if you're NZ Health, giving advice on [how to make meth](https://www.rnz.co.nz/news/national/590723/emergency-department-ai-gives-meth-recipe-in-jailbreak-testing)). They only want relevant questions being passed to the LLM.

The solution to this is to put a guardrail in, which returns a classification before the UI sends the question to the LLM for processing. If the question or query is not related to the main subject area that you cover, then you can reject the user request.

Luckily, you've got a master service agreement with an overpriced consultancy group, who have told you that they can set this up, no worries, for a huge amount of $$$$. Apparently it's very complicated, but they have a top team of experts who can do this for you.

But even more luckily for you, it's not that complicated, and it shouldn't be expensive. In fact, you can set it up on a personal laptop! So tell those consultants to get stuffed.

## What is a guardrail?

A guardrail acts as a very simple yes/no check to make sure that a user request is appropriate. 

For example, if you only want to answer questions about cars, then the guardrail will be trained on car-related questions. If someone asks a question about sports, then it will return a response saying 'not related'.

Because the guardrail process is very simple, you **do not** need a super-powerful LLM to run this. In fact, you can use a lightweight 1.7B parameter model and get perfect results. You can even use a very lightweight model (less than 1 billion parameters) and it will still work and be lightning fast (although the quality starts to drop).

For this project, I'll be using a 1.7B Qwen3 model with Star Trek training data, but you can easily swap this out for anything you prefer. I'll also show you how to set up your training questions.

>The training script uses LoRA (Low-Rank Adaption) fine-tuning, LoRA is a parameter-efficient fine-tuning (PEFT) technique that adapts large pre-trained models to specific tasks by training only a tiny fraction of their parameters. Instead of updating all billions of weights in a model - which is slow and memory-intensive - LoRA "freezes" the original weights and adds small, trainable "adapter" matrices to specific layers.

# Setup steps

## Requirements

- An internet connection
- Administrator access to your computer
- A Hugging Face account (https://huggingface.co/)

All or some of these python modules:

```bash
pip3 install -U "huggingface_hub"
pip3 install datasets
pip3 install peft
pip3 install bitsandbytes
pip3 install flask
pip3 install flask_cors
pip3 install protobuf
```

## Hugging Face

To make sure Hugging Face is working correctly, try this command:

```bash
hf
```

You should see a list of available commands.

To interact with the Hugging Face Hub (in our case, to download and upload models), you will need to log in with an access token. 

- **Generate a Token**: Go to the [Hugging Face Access Tokens settings page](https://huggingface.co/settings/tokens) to create a new access token. You will need read and write access.
- **Log in via CLI**: Run the login command in your terminal and follow the prompts to enter your token:

```bash
hf auth login
```

# Create the model

There are two parts to this process.

- First, you need to train (finetune) an LLM with your guardrail questions
- Second, you set up a basic server instance which uses your new model and use a chat interface to connect to it.

## Part 1: Train an LLM

This example will be using Star Trek training questions, so navigate to the Star Trek directory:

```bash
cd finetuning/star_trek
```

### Step 1: Generate the questions

You need a large set of questions or statements with the appropriate respones so the model can get a feel for what is related or not.

The base Qwen model that we're using already has rich semantic understanding of everything - maths, geography, Star Trek, Star Wars, etc. Fine-tuning adjusts the model's internal representations to draw a decision boundary. The classifier head then sits on top of those representations.

**The general rule:**

- Diverse unrelated examples → good generalisation for unseen unrelated topics
- Narrow unrelated examples → gaps where the model misclassifies unfamiliar inputs

You don't need to enumerate every possible unrelated question, but you do need the unrelated set to be representative of the types of questions that might appear at the boundary. For a Star Trek guard, that means:

- General knowledge
- Math / factual trivia
- Crossover adversarial questions that mention Star Trek entities but aren't about Star Trek

The adversarial crossover examples are the most important addition - without them, the model learns "questions mentioning Vulcans = related", which could be wrong.

Run this script:

```bash
python3 ./generate_dataset.py
```

This will create a file called `guard_dataset.jsonl` - you need to give it a quick look to make sure that the questions seem relevant. 

If you have customised this for a different topic, you'll also need to make sure that it hasn't gone into a repetitive loop.

### Step 2: Train the model

Run this script:

```bash
python3 ./train_model_guard.py
```

By default this will use Qwen3-1.7B. You can use any model that you prefer from Hugging Face.

Qwen is pretty good - the 0.6B model will be very fast, but you'll get better results from the 1.7B model.

**THIS SCRIPT WILL TAKE A LONG TIME**

The more questions you train it on, the slower it takes. I'm using 8,000 questions here - if your topic is very niche then this might be too many, but for a broad topic it should be easy.

The output will be in the `finetuned` directory. You don't need to do anything with these files.

### Step 3: Upload to Hugging Face

Technically you don't need to do this, but the alternatives will require you to build the model locally and move the resulting file around which can be hard work, especially if you've used a multi-billion parameter model. Uploading it to Hugging Face is nice and tidy, _but you will need a Hugging Face account_.

Run the upload script, and provide your username and preferred repository name for your new guardrail model. If the repository name doesn't exit, it will be automatically created.

By default, your model will be private and not visible to anyone else.

```bash
python3 ./huggingface_upload.py
```

Congratulations! You have now created a guardrail model!

## Part 2: Create a chat interface

Now you can put this guardrail into a fully working demo.

Go back to the main directory, and run this command:

```bash
python3 ./api_server.py
```

You'll need to provide the model name you want to use, which is presumably the model name we uploaded to Hugging Face in the previous step. It wll look something like `[your_username]/[model_name]` which you can also see in the Hugging Face URL for your model repository.

This will create a very basic server on port 8080.

If you load `chat.html` into a browser, you'll be able to ask questions and get a classification result.

If you are running this on a laptop then it is likely to be quite slow, but you'll still be able to see the results.

# Next steps

## How to build your own topic

Ok, so this is working perfectly, but you don't want Star Trek questions, in fact you want car related content. 

Generating your own question set is easy. In Cursor (or Claude, or whatever your preferred LLM is), open the `generate_dataset.py` file, add it to a Cursor chat and give this prompt:

_"@generate_dataset.py Update this file to generate questions about cars. Make sure that all the sections are now car-related, and there are no remaining Star Trek references. 
The adversarial section needs appropriate questions as well."_

You can obviously change this text, and you might need to adjust it depending on your new topic and how well the AI agent responds.

### Test questions

You'll also need to update the questions in the `finetune_test.txt` file.

_"@finetune_test.txt Update this file with a list of test questions covering both  car-related and unrelated topics, as well as adversarial questions."

When you train the model using your new `guard_dataset.jsonl` file, you'll see the results from your test questions and hopefully they'll be classified correctly.

Just run the `train_model_guard.py` script again and you'll have your own customised guardrail model!

## Why do some questions get allowed when they're not related?

If this happens, then the unrelated training set was too narrow in character. The model had never seen anything in that region of the embedding space labelled not_related, so it defaulted to the wrong side. You can fix this by adding more unrelated or adverserial questions concerning the unrelated topic.

## How to host the LLM on a better server

If you open the `chat.html` file, you can change the value on line 459:

```javascript
const MODERATION_API_URL = 'http://localhost:8080/api/moderate';
```

Change `localhost` to be the IP address of a more powerful server that you have run the `api_server.py` script on.

For an even better demonstration, you can install Ollama, either on your laptop or another server. Change these values on lines 463 - 467 to reflect your configuration, and you'll be able to actually ask moderated questions AND get an answer!

```javascript
const LLM_CONFIG = {
    baseUrl: 'http://localhost:11434',
    model: 'Qwen3-4B-f16:Q5_K_M',
    endpoint: '/api/chat'
};
```

## How to put this into production

**This example chat interface is clearly NOT production-ready**

The guardrail LLM is perfectly suitable for production use - don't let anyone tell you otherwise. If it passes your quality assuarance tests, then you can use it anywhere, just like a regular LLM.

You'll need to integrate it into whatever actual user interface you have in mind, which might be more complicated though.

## Select an LLM

So far this has been tested on:

- Qwen3-0.6B
- Qwen3-1.7B
- Qwen3-4B
- Qwen3-8B






