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
#core function for ClinicalBERT
    tokenized_sentences, all_spans, normalized_text = parse_text(text)
    all_annotations = clinical_concept_extraction(tokenized_sentences)
    ent = build_display_elements(tokenized_sentences, all_annotations, all_spans)

    term_list = []

    for e in ent:
        text = normalized_text[e['start']:e['end']]
        if e['label'] == 'problem':
            term_list.append(text + "#Problem" + "\n")
        elif e['label'] == 'test':
            term_list.append(text + "#Test" + "\n")
        elif e['label'] == 'treatment':
            term_list.append(text + "#Treatment" + "\n")
            
    return ' '.join(filter(lambda x: x if x is not None else '', term_list))

def indent(elem, level=0):
    i = "\n" + level*"  "
    if len(elem):
        if not elem.text or not elem.text.strip():
            elem.text = i + "  "
        if not elem.tail or not elem.tail.strip():
            elem.tail = i
        for elem in elem:
            indent(elem, level+1)
        if not elem.tail or not elem.tail.strip():
            elem.tail = i
    else:
        if level and (not elem.tail or not elem.tail.strip()):
            elem.tail = i

import xml.etree.ElementTree as ET
#f_path = "/home2/ukyoung/my-python/TREC/문서전처리_0712/CTs-processed-v1/"

doc_path = sys.argv[1]

with open(doc_path, 'r', encoding='UTF8') as i_f:
    doc = ET.parse(i_f)
    doc = doc.getroot()

    # Document ID
    nct_id = str(doc.find('nct_id').text)

    # Titles
    title = str(doc.find('brief_title').text)
    title = get_clinicalTerms(title)
    
    # official_title
    official_title = str(doc.find('official_title').text)
    official_title = get_clinicalTerms(official_title)

    # Summaries
    summary = str(doc.find('brief_summary').text)
    summary = get_clinicalTerms(summary)

    # Descriptions
    description = str(doc.find('detailed_description').text)
    description = get_clinicalTerms(description)
    
    # condition
    condition = str(doc.find('condition').text)
    condition = get_clinicalTerms(condition)

    inclusion_criteria = str(doc.find('inclusion_criteria').text)
    inclusion_criteria = get_clinicalTerms(inclusion_criteria)

    exclusion_criteria = str(doc.find('exclusion_criteria').text)
    exclusion_criteria = get_clinicalTerms(exclusion_criteria)

root = ET.Element('Doc')
root.set('nct_id', nct_id)
ET.SubElement(root, 'title').text = title
ET.SubElement(root, 'official_title').text = official_title
ET.SubElement(root, 'summary').text = summary
ET.SubElement(root, 'description').text = description
ET.SubElement(root, 'condition').text = condition
ET.SubElement(root, 'inclusion_criteria').text = inclusion_criteria
ET.SubElement(root, 'exclusion_criteria').text = exclusion_criteria

indent(root)
ET.dump(root)

tree = ET.ElementTree(root)
tree.write('./outputs/' + nct_id + '.xml', encoding='UTF8')