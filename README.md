# How to set up a guardrail LLM

This project is to demonstrate how to set up a guardrail project that only allows particular question types to reach an LLM.

## Why is this important?

Let's say you're employed by a government agency or a company who is very excited about this whole 'AI thing', but the security or legal team are very worried about the chatbot going rogue with answers about Uncle Elon or el Presidente Trump (or if you're NZ Health, giving advice on [how to make meth](https://www.rnz.co.nz/news/national/590723/emergency-department-ai-gives-meth-recipe-in-jailbreak-testing)). They only want relevant (and appropriate) questions being passed to the LLM.

The solution to this is to put a guardrail in, which returns a classification before the UI sends the question to the LLM for processing. If the question or query is not related to the main subject area that you cover, then you can reject the user request.

Luckily, you've got a master service agreement with an overpriced consultancy group, who have told you that they can set this up, no worries, for a huge amount of $$$$. Apparently it's very complicated, but they have a top team of experts who can do this for you.

But even more luckily for you, it's not that complicated, and it shouldn't be expensive. In fact, you can set it up on a personal laptop! So tell those consultants to get stuffed.

## What is a guardrail?

A guardrail acts as a very simple yes/no check to make sure that a user request is appropriate. 

For example, if you only want to answer questions about cars, then the guardrail will be trained on car-related questions. If someone asks a question about sports, then it will return a response saying 'not related'.

The key idea is that the guardrail is a **separate, lightweight model** that sits in front of your main LLM. The flow looks like this:

1. The user submits a question
2. The guardrail model classifies it as **related** or **not related** (this is fast - it's a tiny model doing a simple classification)
3. If the question is related, it gets forwarded to the main LLM for a full answer
4. If it's not related, the request is rejected before it ever reaches the main LLM

Because the guardrail is just a classifier, you **do not** need a super-powerful LLM to run it. In fact, you can use a lightweight 1.7B parameter model (Qwen3-1.7B) and get perfect results. You can even use a very lightweight model (less than 1 billion parameters such as Qwen3-0.6B) and it will still work and be lightning fast (although the quality starts to drop).

For this project, I'll be using a 1.7B Qwen3 model with Star Trek training data (that we'll create here), but you can easily swap this out for anything you prefer. I'll also show you how to set up your training questions.

# Setup steps

## Requirements

- An internet connection
- Administrator access to your computer
- A Hugging Face account (https://huggingface.co/)
- Python 3.x

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

- First, you need to train (finetune) an LLM with your guardrail questions.
- Second, you set up a basic server instance which uses your new model and use a chat interface to connect to it.

## Part 1: Train an LLM

This example will be using Star Trek training questions, so navigate to the Star Trek directory:

```bash
cd finetuning/star_trek
```

### Step 1: Generate the questions

You need a large set of questions or statements with the appropriate responses so the model can get a feel for what is related or not.

The base Qwen model that we're using already has rich semantic understanding of everything - maths, geography, Star Trek, Star Wars, etc. Fine-tuning teaches the model to draw a boundary between related and not-related inputs, so it can confidently classify questions it hasn't seen before.

**The general rule:**

- Diverse unrelated examples → good generalisation for unseen unrelated topics
- Narrow unrelated examples → gaps where the model misclassifies unfamiliar inputs

You don't need to enumerate every possible unrelated question, but you do need the unrelated set to be representative of the types of questions that might appear at the boundary. For a Star Trek guard, that means:

- General knowledge
- Math / factual trivia
- Crossover adversarial questions that mention Star Trek entities but aren't about Star Trek

The adversarial crossover examples are the most important addition - without them, the model learns that all "questions mentioning Vulcans = related", which could be wrong.

Run this script:

```bash
python3 ./generate_classification_dataset.py
```

This will create a file called `classification_dataset.jsonl` - you need to give it a quick look to make sure that the questions seem relevant. Each line is a JSON object with a question and a label:

```json
{"input": "What engine does the Enterprise use?", "label": "related"}
{"input": "What is the capital of France?", "label": "not_related"}
```

If you have customised this for a different topic, you'll also need to make sure that it hasn't gone into a repetitive loop. This sometimes happens if your topic is very niche and the number of questions you've asked for exceeds the likely knowledge base of the LLM. In these cases, try reducing the number of questions.

### Step 2: Train the model

Run this script:

```bash
python3 ./train_classification_model.py
```

By default this will use Qwen3-1.7B. You can use any model that you prefer from Hugging Face. So far this has been tested on:

| Model | Speed | Accuracy | Notes |
|---|---|---|---|
| Qwen3-0.6B | Very fast | Good | Best for low-resource environments, but accuracy drops on edge cases |
| Qwen3-1.7B | Fast | Excellent | Recommended starting point — best balance of speed and quality |
| Qwen3-4B | Moderate | Excellent | Diminishing returns over 1.7B for a simple classifier |
| Qwen3-8B | Slow | Excellent | Overkill for guardrail use, but works well if you have the hardware |

**THIS SCRIPT WILL TAKE A LONG TIME**

The more questions you train it on, the slower it takes. I'm using 8,000 questions here - if your topic is very niche then this might be too many, but for a broad topic it should be easy.

The output will be in the `finetuned` directory. You don't need to do anything with these files.

>**What is LoRA?** The training script uses LoRA (Low-Rank Adaptation) fine-tuning. LoRA is a parameter-efficient fine-tuning (PEFT) technique that adapts large pre-trained models to specific tasks by training only a tiny fraction of their parameters. Instead of updating all billions of weights in a model - which is slow and memory-intensive - LoRA "freezes" the original weights and adds small, trainable "adapter" matrices to specific layers. This is why you can train a guardrail model on a personal laptop.

### Step 3: Upload to Hugging Face

Technically you don't need to do this, but the alternatives will require you to build the model locally and move the resulting file around which can be hard work, especially if you've used a multi-billion parameter model. Uploading it to Hugging Face is nice and tidy, _but you will need a Hugging Face account_.

Run the upload script, and provide your username and preferred repository name for your new guardrail model. If the repository name doesn't exist, it will be automatically created.

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

You'll need to provide the model name you want to use, which is presumably the model name you uploaded to Hugging Face in the previous step. It will look something like `[your_username]/[model_name]` which you can also see in the Hugging Face URL for your model repository.

This will create a very basic server on port 8080.

If you load `chat.html` into a browser, you'll be able to ask questions and get a classification result.

If you are running this on a laptop then it is likely to be quite slow, but you'll still be able to see the results.

# Customisation

## How to build your own topic

Ok, so this is working perfectly, but you don't want Star Trek questions, in fact you want car related content. 

Generating your own question set is easy. In Cursor (or Claude, or whatever your preferred LLM is), open the `generate_classification_dataset.py` file, add it to a Cursor chat and give this prompt:

_"@generate_classification_dataset.py Update this file to generate questions about cars. Make sure that all the sections are now car-related, and there are no remaining Star Trek references. 
The adversarial section needs appropriate questions as well."_

You can obviously change this text, and you might need to adjust it depending on your new topic and how well the AI agent responds.

## Test questions

You'll also need to update the questions in the `finetune_test.txt` file.

_"@finetune_test.txt Update this file with a list of test questions covering both car-related and unrelated topics, as well as adversarial questions."_

When you train the model using your new `classification_dataset.jsonl` file, you'll see the results from your test questions and hopefully they'll be classified correctly.

Just run the `train_classification_model.py` script again and you'll have your own customised guardrail model!

## Appropriateness vs relatedness

Being on-topic doesn't automatically mean a question is appropriate. _"How do I disable a car's GPS tracker?"_ is entirely car-related. _"What's the lethal dose of paracetamol?"_ is a healthcare question. Both would pass the topic guardrail, but neither is something you'd want to answer without careful handling.

### The recommended approach: rely on the main LLM

Modern aligned models (Qwen3, Llama 3, etc.) have safety training built in and will refuse harmful or inappropriate requests without any extra configuration. This is far more reliable than trying to train a small classifier to judge intent — small models aren't good at nuance.

The most effective setup is also the simplest:

1. Use the topic guardrail to filter out off-topic questions cheaply and quickly.
2. Let a well-configured main LLM handle everything that gets through.

A clear system prompt covers most edge cases with no extra complexity:

> _"You are a [domain] assistant. Answer questions about [topic] only. Do not provide [specific exclusions], even hypothetically."_

This gives you cost savings (off-topic questions never reach the more expensive LLM), better safety judgement (you're relying on a model actually trained for it), and less to maintain.

### If you need an explicit refusal layer

If the main LLM's built-in safety isn't sufficient — for example, you're running a base model without alignment tuning, or you're in a compliance environment that requires explicit, auditable filtering — you can add a refusal layer between the topic guardrail and the main LLM.

The flow becomes:

1. The topic guardrail checks: is this related? If not, reject it.
2. The refusal layer checks: is this appropriate? If not, reject it with a specific message.
3. If both checks pass, the question goes to the main LLM.

There are three approaches to implementing the refusal layer, in order of complexity:

**Rule-based filters** are the simplest option — a list of blocked keywords or phrases that automatically trigger a rejection. They're fast, transparent, and easy to audit, which compliance teams appreciate. The downside is brittleness: they'll miss anything not on the list, and can catch legitimate questions that happen to contain a flagged word.

**A second classifier model** works the same way as the topic guardrail, but trained on examples of appropriate versus inappropriate questions within your domain. This is what the included scripts implement. The `train_refusal_model.py` script trains a multi-label classifier that returns a specific refusal category (`weapons`, `privacy`, `piracy`, `explicit`, `medical`, `harmful`, `self_harm`) so the UI can return a tailored message. This works well when you have a clear, stable policy about what "inappropriate" means. If your policy is ambiguous or likely to evolve, you'll spend more time retraining it than it saves.

**Semantic blocking** sits between the two: you define a set of banned concepts and use similarity matching to catch questions that are semantically close to those concepts, even if they don't use the exact words. More flexible than keyword lists, less maintenance-heavy than a trained model.

The main advantage of any explicit refusal layer over relying purely on the main LLM is **auditability** — you can log exactly what was refused and why, with a clear policy reference. That's often what compliance and legal teams need, regardless of how the main LLM would have handled it anyway.

To add the second-classifier refusal layer, use the included scripts:

- `generate_refusal_dataset.py` — generates labelled training examples for each refusal category
- `train_refusal_model.py` — fine-tunes the multi-label classifier

# Troubleshooting

## Why do some questions get allowed when they're not related?

If this happens, then the unrelated training set was too narrow in character. The model had never seen anything in that region of the embedding space labelled _not_related_, so it defaulted to the wrong side. You can fix this by adding more unrelated or adversarial questions concerning the unrelated topic.

# Deployment

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

The guardrail LLM is perfectly suitable for production use - don't let anyone tell you otherwise. If it passes your quality assurance tests, then you can use it anywhere, just like a regular LLM.

You'll need to integrate it into whatever actual user interface you have in mind, which might be more complicated though.

