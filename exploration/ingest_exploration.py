
# just for exploration, final solution use langchain pinecone lib integration
# openai implementation to create embedding and save it not feasible
# take too much time for embedd each data, 2.5 iter/sec

#%%
import pandas as pd
from tqdm import tqdm
tqdm.pandas()

# from pandarallel import pandarallel
# pandarallel.initialize(progress_bar=True, nb_workers=-1)

df=pd.read_csv('../data/SPOTIFY_REVIEWS_CLEAN.csv')

#%%
df_mini = df.iloc[:100]
df_mini.to_csv('../data/data_mini.csv', index=False)

#%%
df_mini.review_text[4]

#%%
text = df_mini.review_text[4]

#%%
from openai import OpenAI
client = OpenAI()

def get_embedding(text, model="text-embedding-3-small"):
   text = text.replace("\n", " ")
   return client.embeddings.create(input = [text], model=model).data[0].embedding

# df['embedding'] = df.review_text.apply(lambda x: get_embedding(x, model='text-embedding-3-small'))
print('embedding process.....')
df['embedding'] = df.review_text.parallel_apply(lambda x: get_embedding(x, model='text-embedding-3-small'))
df.to_csv('data/embedded_spotify_reviews.csv', index=False)

#%% test satu satu
from openai import OpenAI
client = OpenAI()
saver = client.embeddings.create(input = [df['review_text'][4]], model="text-embedding-3-small", dimensions=256)

#%%
len(saver.data[0].embedding)

#%%
from langchain import hub
retrieval_qa_chat_prompt = hub.pull("langchain-ai/retrieval-qa-chat")

for i in retrieval_qa_chat_prompt:
   print(i)
   
# %%
