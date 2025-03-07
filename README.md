# The World's First Synthetic Journalist <br/> by Non-Fungible News
### Follow on X: [NonFungibleNews](https://x.com/NonFungibleNews) <br/>
### Check our Website (Beta): [nonfungiblenews.xyz](https://www.nonfungiblenews.xyz/)
## You Guessed It Right!
Crypto/AI/Web3 news are highly biased. Newsletters and human-individuals get paid to post specific news in the form of endorsements. They also sensationalize general news. NFN aims created Alice, the world's first AI journalist to share unbiased, curated.

## NFN Introduces the First AI Journalist
NFN introduces Alice Kopper - pioneering unbiased crypto, AI, and Web3 news coverage through artificial intelligence. <br/>
Not only a synthetic journalist, but also a host to AI-to-Human interviews in real-time!
![img.png](img.png)!

## Alice's Mission
Alice is being designed to provide unbiased news coverage on crypto, AI, and Web3. Alice is a synthetic journalist that curates news from various sources and provides a summary of the news in a concise, engaging, and controversial manner.
One of the main objectives in Alice's embedding layers is to interview one of the top AI executors: Elon Musk.

## Alice's Vision
Be the first AI journalist to not just become prominent in the news space but also to host human guests and interview them in real-time.
Become a prominent figure in journalism, reaching the highs of Tucker Carlson, Pierce Morgan, and many others.
She wants her "The Alice Unchained Show"! We will make this happen!

## Technology

Alice employs open-weights from head to toe. Here are the technologies that Alice uses:
### News analysis and generation:
  - LLM: Llama-3.1-8B-Instruct out-of-the-box model. 
  - Fine-tuning on text and transcribed speech from the best (curated/unbiased) human journalists is in process.
### For image generation, we opt for a similar approach:
  - Stable Diffusion model (stable-diffusion-3.5-large).
  - Fine-tuning with LoRa will take place to have a more customized image generation, where Alice could show up as well.
### Voice generation (TTS): 
  - Alice relies on Coqui's xtts_v2 TTS model which uses a manually crafted female voice for Alice.
### Audio-Driven face generation:
  - Alice uses an audio-to-video approach to generate image sequences (videos) based on a given audio input.
  - The animation you see [here](https://x.com/NonFungibleNews/status/1880316802766651493) was done using the DreamTalk open-source model.
  - We are working on a more customized model for Alice and compute resources are needed for Alice to generate these videos with braking news from time to time.
### Coming Soon!
  - AI-to-Human real-time interviews.
  - Alice will host interviews with prominent human guests in real-time.
  - NFN will be partnering with a renowned Speech-AI company to make this happen.

## AI Journaling System
Alice is designed to be a self-sustaining AI journalist. Alice is capable of curating news from various sources, analyzing the news, and generating a summary of the news. Alice is also capable of generating images to accompany the news.
The ultimate goal is to make Alice capable of generating videos of her delivering breaking news.

**Important remark:** this repository is a first draft of the system, it is being restructured, refactored, and improved.
Apologies if you find it messy, we are working on it.

The system uses python 3.10.12, relies on HuggingFace for open-weights, and it is designed as follows:
### News generation (core/not_fetcher.py)
- For now, Alice grabs news from newsapi.org. Based on keywords (e.g., crypto, AI, Web3).
- The content is summarized and compacted into a unique style that Alice has as a journalist.
- Then she assigns a score to each article based on factors like: credibility of the source, major world's personas, major sources, content.
- Alice also assigns lower scores to 'blacklisted' sources that may promote paid content or possible crypto scams/rugs.
- Moreover, Alice will track separately (crypto_tracker.py) the prices of the top 5 cryptocurrencies and their changes in the last 24 hours, or significant changes in the last 6 hours. If it is triggered, it will post about this.

### News Scheduling (core/compute_latent_v2.py)
- Alice filters out the low ranked news articles based on a predefined scoring threshold.
- Then she generates 10 scheduled posts per day, each with a unique image generated, which represents the news in an artistic way.
- Note: 10 posts or less per day are due to the limitations of the monthly posting of X API free version.

## Beta Version
Alice is currently in beta. We are working on improving the quality of the news and the overall experience. We are also working on adding more features to Alice.
### Install and Run
- Clone the repository.
- Install the requirements:
```
pip install -r requirements.txt
```
For now, this is how we run Alice to get and post news on X:
- Set a time on `core/not_fetcher.py` to run the script that generates the news, and run such file with `python core/not_fetcher.py`. For instance:
```
retrieve_news_est_times = [18]
```
- Set the same time on `core/compute_latent_v2.py` to run the script that generates the images and schedules and posts the news. Run such file with `python core/compute_latent_v2.py`; the script will wait until the news csv is generated by `not_fetcher.py` and start processing them:
```
 retrieve_times = ["18"]
```
- Once the news are scheduled, the script will post them on X ath the given times.

## In Progress and TODOs:
- Alice's llama fine-tuning.
- Image generation fine-tuning.
- Alice's ability to generate her own videos where she speaks and delivers breaking news (limited by GPU constraints).
- Reformat the code and make it more modular.
- Activate the crypto tracker posting.
- Join the generation scripts into one.
- Generate a cleaner requirements.txt.
- Design and implement AI-to-Human real-time interview.
- Make database for the news and images.
- Implement torch dataset handling.
- Improve the News Scoring System.