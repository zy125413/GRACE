from transformers import  LlamaTokenizer
from argparse import ArgumentParser
from tqdm import trange
import random
from tools import read_predata,tokenize_predata
import pickle
def random_chunk_shuffle(text):
    """
    Shuffle the text by splitting it into random chunks of 1-15 words and then shuffling these chunks.
    """
    words = text.split()
    chunks = []
    while words:
        try:
            chunk_size = random.randint(5, min(15, len(words)))  # Choose a random chunk size between 1 and 15
            chunk = words[:chunk_size]                          # Get the next chunk
            words = words[chunk_size:]                          # Remove this chunk from the list
            chunks.append(' '.join(chunk))                      # Add the joined chunk to the chunks list
        except:
            chunk_size=len(words)
            chunk = words[:chunk_size]                          # Get the next chunk
            words = words[chunk_size:]                          # Remove this chunk from the list
            chunks.append(' '.join(chunk))

    
    random.shuffle(chunks)  # Shuffle the chunks
    return ' '.join(chunks)  # Join the shuffled chunks into a string


def hyper_parameters():
    parser = ArgumentParser(description='GPT2')

    parser.add_argument('--model_dir', type=str, default='./llama-2-7b-hf')
    parser.add_argument('--tokenizer_dir', type=str, default='./llama-2-7b-hf')
    parser.add_argument('--name', type=str, default='0')
    parser.add_argument('--dir', type=str,default='./stackexchange.json')
    parser.add_argument('--out_dir', type=str,default='stackexchange.pkl')
    parser.add_argument('--out_random_dir', type=str,default='random_stackexchange.pkl')
    parser.add_argument('--n', type=int,default=2000)
    opt = parser.parse_args()
    return opt


if __name__ == '__main__':    
    hps = hyper_parameters()
    tokenizer = LlamaTokenizer.from_pretrained(hps.tokenizer_dir)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    datas = read_predata(hps.dir)
    text=[]
    text_random=[]
    for i in range(len(datas)):
        text.append(datas[i]['text'])
        text.append(random_chunk_shuffle(datas[i]['text']))
    text_tk=tokenize_predata(text[:hps.n], tokenizer)
    text_random_tk=tokenize_predata(text[:hps.n], tokenizer)
    with open (hps.out_dir,"wb") as f:
        pickle.dump(text_tk,f)
    with open (hps.random_out_dir,"wb") as f:
        pickle.dump(text_random_tk,f)

