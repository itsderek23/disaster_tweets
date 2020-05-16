#!/usr/bin/env python
# coding: utf-8
import disaster_tweets
from disaster_tweets.util.notebook_imports import *
from disaster_tweets.util.dataset import *
import whisk


# ## Data Cleaning
# As we know,twitter tweets always have to be cleaned before we go onto modelling.So we will do some basic cleaning such as spelling correction,removing punctuations,removing html tags and emojis etc.So let's start.

# In[7]:


tweet, test = dataset_to_df()
df=pd.concat([tweet,test])
df.shape


# ### Removing urls

# In[8]:


example="New competition launched :https://www.kaggle.com/c/nlp-getting-started"


# In[9]:


def remove_URL(text):
    url = re.compile(r'https?://\S+|www\.\S+')
    return url.sub(r'',text)

remove_URL(example)


# In[10]:


df['text']=df['text'].apply(lambda x : remove_URL(x))


# ### Removing HTML tags

# In[11]:


example = """<div>
<h1>Real or Fake</h1>
<p>Kaggle </p>
<a href="https://www.kaggle.com/c/nlp-getting-started">getting started</a>
</div>"""


# In[12]:


def remove_html(text):
    html=re.compile(r'<.*?>')
    return html.sub(r'',text)
print(remove_html(example))


# In[13]:


df['text']=df['text'].apply(lambda x : remove_html(x))


# ### Romoving Emojis

# In[14]:


# Reference : https://gist.github.com/slowkow/7a7f61f495e3dbb7e3d767f97bd7304b
def remove_emoji(text):
    emoji_pattern = re.compile("["
                           u"\U0001F600-\U0001F64F"  # emoticons
                           u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                           u"\U0001F680-\U0001F6FF"  # transport & map symbols
                           u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           u"\U00002702-\U000027B0"
                           u"\U000024C2-\U0001F251"
                           "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r'', text)

remove_emoji("Omg another Earthquake 😔😔")


# In[15]:


df['text']=df['text'].apply(lambda x: remove_emoji(x))


# ### Removing punctuations

# In[16]:


def remove_punct(text):
    table=str.maketrans('','',string.punctuation)
    return text.translate(table)

example="I am a #king"
print(remove_punct(example))


# In[17]:


df['text']=df['text'].apply(lambda x : remove_punct(x))


# ## GloVe for Vectorization

# Here we will use GloVe pretrained corpus model to represent our words.It is available in 3 varieties :50D ,100D and 200 Dimentional.We will try 100 D here.

# In[18]:



def create_corpus(df):
    corpus=[]
    for tweet in tqdm(df['text']):
        words=[word.lower() for word in word_tokenize(tweet) if((word.isalpha()==1) & (word not in stop))]
        corpus.append(words)
    return corpus




# In[19]:


corpus=create_corpus(df)


# In[20]:


embedding_dict={}
with open('data/raw/glove.6B.100d.txt','r') as f:
    for line in f:
        values=line.split()
        word=values[0]
        vectors=np.asarray(values[1:],'float32')
        embedding_dict[word]=vectors
f.close()


# In[21]:


MAX_LEN=50
tokenizer_obj=Tokenizer()
tokenizer_obj.fit_on_texts(corpus)
sequences=tokenizer_obj.texts_to_sequences(corpus)

tweet_pad=pad_sequences(sequences,maxlen=MAX_LEN,truncating='post',padding='post')


# In[22]:


word_index=tokenizer_obj.word_index
print('Number of unique words:',len(word_index))


# In[23]:


num_words=len(word_index)+1
embedding_matrix=np.zeros((num_words,100))

for word,i in tqdm(word_index.items()):
    if i > num_words:
        continue

    emb_vec=embedding_dict.get(word)
    if emb_vec is not None:
        embedding_matrix[i]=emb_vec



# In[24]:


# pd.to_csv("data/processed/em", header=None, index=None)
tweet_pad[0]


# ## Baseline Model

# In[25]:


model=Sequential()

embedding=Embedding(num_words,100,embeddings_initializer=Constant(embedding_matrix),
                   input_length=MAX_LEN,trainable=False)

model.add(embedding)
model.add(SpatialDropout1D(0.2))
model.add(LSTM(64, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(1, activation='sigmoid'))


optimzer=Adam(learning_rate=1e-5)

model.compile(loss='binary_crossentropy',optimizer=optimzer,metrics=['accuracy'])


# In[26]:


model.summary()


# In[28]:


train=tweet_pad[:tweet.shape[0]]
test=tweet_pad[tweet.shape[0]:]
print(test.shape)


# In[ ]:


X_train,X_test,y_train,y_test=train_test_split(train,tweet['target'].values,test_size=0.15)
print('Shape of train',X_train.shape)
print("Shape of Validation ",X_test.shape)


# In[ ]:


history=model.fit(X_train,y_train,batch_size=4,epochs=15,validation_data=(X_test,y_test),verbose=2)


# ## Save the model and tokenizer

# In[ ]:


model.save(whisk.artifacts_dir / "model.h5")
with open(whisk.artifacts_dir / 'tokenizer.pickle', 'wb') as handle:
    pickle.dump(tokenizer_obj, handle, protocol=pickle.HIGHEST_PROTOCOL)
