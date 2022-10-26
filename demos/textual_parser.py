from argparse import _SubParsersAction
import re
import pysrt
from bs4 import BeautifulSoup
from nltk.tokenize import RegexpTokenizer
from sentence_transformers import SentenceTransformer, util
import numpy as np
SEARCH_THRESHOLD = 5

model = SentenceTransformer('stsb-roberta-large')
def remove_utf_symbols(text):
    chars = {
        b'\xc2\x82': b',',  # High code comma
        b'\xc2\x84': b',,',  # High code double comma
        b'\xc2\x85': b'...',  # Triple dot
        b'\xc2\x88': b'^',  # High carat
        b'\xc2\x91': b'\x27',  # Forward single quote
        b'\xc2\x92': b'\x27',  # Reverse single quote
        b'\xc2\x93': b'\x22',  # Forward double quote
        b'\xc2\x94': b'\x22',  # Reverse double quote
        b'\xc2\x95': b' ',
        b'\xc2\x96': b'-',  # High hyphen
        b'\xc2\x97': b'--',  # Double hyphen
        b'\xc2\x99': b' ',
        b'\xc2\xa0': b' ',
        b'\xc2\xa6': b'|',  # Split vertical bar
        b'\xc2\xab': b'<<',  # Double less than
        b'\xc2\xbb': b'>>',  # Double greater than
        b'\xc2\xbc': b'1/4',  # one quarter
        b'\xc2\xbd': b'1/2',  # one half
        b'\xc2\xbe': b'3/4',  # three quarters
        b'\xca\xbf': b'\x27',  # c-single quote
        b'\xcc\xa8': b'',  # modifier - under curve
        b'\xcc\xb1': b''  # modifier - under line
    }

    def replace_chars(match):
        char = match.group(0)
        return chars[char]

    return re.sub(b'|'.join(chars.keys()), replace_chars, text)

""" Function for parsing subtitle dialogues to a set of sentences
    """
def process_subtitles(file):
    subs = pysrt.open(file)
    subtitles = []
    for i in range(0, len(subs)):
        text = subs[i]
        subtitles.append([subs[i].start.to_time(), subs[i].end.to_time(), subs[i].text.replace('\n', ' ')])
    return subtitles

""" Function for parsing scripts dialogues to a set of sentences
    """
def process_script(doc):
    html_doc = open(doc, 'rb')
    soup = BeautifulSoup(html_doc, 'html.parser')
    script = []

    for each in soup.find_all('b'):
        text = each.text.replace('\r', '').replace('\n', '').strip()
        if len(text) >= 3:
            next = each.next_sibling
            if next is None:
                continue
            next = next.get_text()
            next = re.sub(r'<[^>]*>', '', next)
            next = next.encode('utf-8')
            next = remove_utf_symbols(next).decode('utf-8')
            next = re.sub(r'\([^)]*\)', '', next)
            next = re.sub(r'{[^}]*\}', '', next)
            if next.startswith('\r\n'):
                continue
            next = re.split('\r\n\r\n', next)
            next = next[0].replace('\r', '').replace('\n', ' ').strip()
            next = [m.group().strip() for m in re.finditer(r' *((.{0,99})(\.|.$))',next)]
            
            for k in next:
                for p in re.split('\?', k):
                    for n in re.split('\.\.\.', p):
                        n = re.sub(r'\([^)]*\)', '', n)
                        if len(n.strip()) < 1:
                            break
                        n = n.replace("--", '')
                        n = n.replace("'", '')
                        script.append([text, n.strip()])

    return script

""" Function for calculating similarity score
    """
def calculate_similarity_score(sentence1, sentence2):
    embedding1 = model.encode(sentence1, convert_to_tensor=True)
    embedding2 = model.encode(sentence2, convert_to_tensor=True)
    # compute similarity scores of two embeddings
    return util.pytorch_cos_sim(embedding1, embedding2)

""" Function for checking word by word when semantic similarity is ineffective
    Args:
        sentence1 & sentence1 (sentences for comparison)
    Raises:
        ValueError: Thrown when specifying a negative path of required files
    """
def check_word_by_word(sentence1, sentence2):
    tokenizer = RegexpTokenizer(r'\w+')
    subs_words = tokenizer.tokenize(sentence1)
    script_words = tokenizer.tokenize(sentence2)
    for i in range(len(subs_words)):
        for j in range(len(script_words)):
            if len(subs_words) - i == 1 and len(script_words) - j >= 1:
                if subs_words[i].lower() == script_words[j].lower():
                    return True
            else:
                if len(subs_words) - i == 2 and len(script_words) - j >= 2:
                    if subs_words[i].lower() == script_words[j].lower() and subs_words[i+1].lower() == script_words[j+1].lower():
                        return True
                else:
                    if len(subs_words) - i >= 3 and len(script_words) - j >= 3:
                        if subs_words[i].lower() == script_words[j].lower() and subs_words[i+1].lower() == script_words[j+1].lower():
                            if subs_words[i+2].lower() == script_words[j+2].lower():
                                return True
    return False

""" Function for wrapping textual annotations
    Args:
        subtitles ( A list of subtitles dialogues)
        scripts ( A list of scripts dialogues)
    """
def wrapping_textual_annotations(subtitles, scripts):
    final_list = []
    search_j_threhold = 20
    search_i_threhold = 3
    index_j = 0
    sentence_matched = 0
    count_i = 0
    for i in range(len(subtitles)):
        count_j = 0
        for j in range(index_j, len(scripts)):
            count_j+=1
            score = calculate_similarity_score(subtitles[i][2], scripts[j][1])
            if score.item() > 0.5:
                sentence_matched+=1
                final_list.append([subtitles[i][0], subtitles[i][1], scripts[j][0]])
                if count_i > search_i_threhold:
                    if j >= search_i_threhold:
                        index_j=j - search_i_threhold
                    else:
                        index_j = j
                    count_i=0
                count_i+=1
                break
            else:
                if check_word_by_word(subtitles[i][2], scripts[j][1]):
                    sentence_matched+=1
                    final_list.append([subtitles[i][0], subtitles[i][1], scripts[j][0]])
                    if count_i > search_i_threhold:
                        if j >= search_i_threhold:
                            index_j=j - search_i_threhold
                        else:
                            index_j = j
                        count_i=0
                    count_i+=1
                    break
                if count_j > search_j_threhold:
                    final_list.append([subtitles[i][0], subtitles[i][1], ""])
                    break

    percent = sentence_matched*100/len(subtitles)
    print("Sentences matched:", percent)
    print("Size subtitles:", len(subtitles))
    print("Size scripts:", len(scripts))
    
    print("-------------------------")
    return final_list

""" Function for getting dialogues in subtitle annotations
    Args:
        subtitle_annos(subtitle annotations of specific movie in MovieNet dataset)
    """
def process_shots(subtitle_annos):
    subtitles = []
    for i in range(len(subtitle_annos)):
        for lines in subtitle_annos[i]['subtitle']:
            for line in lines['sentences']:
                subtitles.append([lines['shot'], lines['shot'], line.replace("'", '')])
    return subtitles

""" Function for textual proccessing
    """
def get_textuals(subtitle_annos, script_file):
    subtitles = process_subtitles(subtitle_annos)
    script = process_script(script_file)
   
    final = wrapping_textual_annotations(subtitles, script)
    return final

def get_label(shot_id, final):
    shots = [ id for id in final if id[1] == shot_id ]
    a = list(shots[:2])
    print(a)
    b = max(a, key = a.count)
    return b[2]