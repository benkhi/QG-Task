import nltk
import re
import os
import sys

from spacy import displacy
from IPython.core.display import display, HTML
from simple_sentence_segment import sentence_segment

from elasticsearch import Elasticsearch
import xml.etree.ElementTree as ET
from tqdm import tqdm
import pandas as pd

def parse_text(text):
    # Perform sentence segmentation, tokenization and return the lists of tokens,
    # spans, and text for every sentence respectively
    tokenizer = nltk.tokenize.TreebankWordTokenizer()
    all_sentences = []
    all_spans = []
    start = 0
    normalized_text = ''
    for span in sentence_segment(text):
        sentence = text[span[0]:span[1]]
        sentence = re.sub('\n', ' ', sentence)
        sentence = re.sub(r'\ +', ' ', sentence)
        sentence = sentence.strip()

        if len(sentence) > 0:
            tokens_span = tokenizer.span_tokenize(sentence)
            tokens = []
            spans = []
            for span in tokens_span:
                tokens.append(sentence[span[0]:span[1]])
                spans.append([start + span[0], start + span[1]])
                
            all_sentences.append(tokens)
            all_spans.append(spans)
            
            start += len(sentence) + 1
            normalized_text += sentence + '\n'
    return all_sentences, all_spans, normalized_text.strip()

def build_display_elements(tokens, annotations, spans):
    # convert the annotations to the format used in displacy
    all_ann = []

    for sent_id, sent_info in enumerate(tokens):
        sent_length = len(tokens[sent_id])

        last_ann = 'O'
        last_start = None
        last_end = None
        for token_id in range(sent_length):
            this_ann = annotations[sent_id][token_id]

            # separated cases:
            if this_ann != last_ann:
                if last_ann != 'O':
                    # write last item
                    new_ent = {}
                    new_ent['start'] = last_start
                    new_ent['end'] = last_end
                    new_ent['label'] = last_ann[2:]
                    all_ann.append(new_ent)

                # record this instance
                last_ann = 'O' if this_ann == 'O' else 'I' + this_ann[1:]
                last_start = spans[sent_id][token_id][0]
                last_end = spans[sent_id][token_id][1]

            else:
                last_ann = this_ann
                last_end = spans[sent_id][token_id][1]

        if last_ann != 'O':
            new_ent = {}
            new_ent['start'] = last_start
            new_ent['end'] = last_end
            new_ent['label'] = last_ann[2:]
            all_ann.append(new_ent)

    return all_ann

#CCE_ASSETS = '/home2/dalya/clinical_concept_extraction/clinical_concept_extraction/cce_assets'
os.chdir('/home2/dalya/clinical_concept_extraction/')

from clinical_concept_extraction import clinical_concept_extraction
os.chdir('/home2/dalya/clinical_concept_extraction/myDir')

# Text -> Term Lists
def get_clinicalTerms(text: str):

    tokenized_sentences, all_spans, normalized_text = parse_text(text)
    all_annotations = clinical_concept_extraction(tokenized_sentences)
    ent = build_display_elements(tokenized_sentences, all_annotations, all_spans)
    
    # see annotations for each tokens
    for sent_, ann_ in zip(tokenized_sentences, all_annotations):
        for e in ent:
            tlist = []
            for t, a in zip(sent_, ann_):
                #print('%30s %s' % (a , t))
                
                text = ('%30s %s' % (t, a) + "\n")
                tlist.append(text)
    return ' '.join(filter(lambda x: x if x is not None else '', tlist))

#f_path = "/home2/ukyoung/my-python/TREC/문서전처리_0712/CTs-processed-v1/"

doc_path = sys.argv[1]

file_index = 0
with open(doc_path, 'r', encoding='UTF8') as i_f:
    doc = i_f.readlines()
    getBIO = get_clinicalTerms(str(doc))
    file_index += 1
    with open('./textBIO/'% file_index + '.txt', 'w') as o_f:
        o_f.write(getBIO)

    
    
