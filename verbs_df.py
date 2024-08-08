import pandas as pd
import spacy
import os
import json
from bs4 import BeautifulSoup

data = {
    "DITA Tags": [
        "apiname", "b", "body", "bodydiv", "chapter", "chdesc", "cmd", "cmdname", 
        "codeblock", "codeph", "coderef", "component", "conbody", "conbodydiv", 
        "concept", "context", "copyrholder", "copyright", "copyryear", "data-about", 
        "data", "desc", "div", "dl", "dlentry", "dlhead", "dt", "entry", 
        "glossAbbreviation", "glossBody", "glossPartOfSpeech", "glossdef", "glossentry", 
        "glossgroup", "glossref", "glossterm", "hazardstatement", "hazardsymbol", 
        "keyword", "keywords", "mainbooktitle", "map", "mathml", "mathmlref", 
        "menucascade", "msgblock", "msgnum", "note", "notices", "ol", "otherinfo", 
        "p", "param", "parameterentity", "parml", "parmname", "part", "ph", "platform", 
        "postreq", "prelreqs", "prereq", "prodinfo", "prodname", "prognum", "prolog", 
        "propdesc", "properties", "refbody", "refbodydiv", "reference", "resourceid", 
        "section", "sectiondiv", "shortdesc", "simpletable", "stentry", "step", 
        "stepresult", "steps-informal", "steps-unordered", "steps", "stepsection", 
        "steptroubleshooting", "stepxmp", "substep", "substeps", "synph", 
        "systemoutput", "table", "task", "taskbody", "tasktroubleshooting", "tbody", 
        "term", "tested", "title", "titlealts", "topic", "topicref", "topicset", 
        "topicsetref", "topicsubject", "troublebody", "troubleshooting", 
        "troubleSolution", "tt", "typeofhazard", "u", "uicontrol", "ul", "userinput", 
        "var", "varname", "volume", "vrmlist", "wintitle", "xmlpi", "xref"
    ],
    "HTML Tags": [
        "<code>", ["<b>", "<strong>"], "<body>", "<div>", 
        "<div>", ["<p>", "<div>"], ["<code>", "<kbd>"], ["<code>", "<span>"], 
        "<pre>", "<code>", "<a>", "<div>", 
        "<div>", "<div>", "<article>", "<div>", "", "<span>", 
        "", "<meta>", ["<data>", "<div>"], "<p>", 
        "<div>", "<dl>", ["<dt>", "<dd>"], "<dt>", "<dt>", ["<td>", "<th>"], "<abbr>", 
        "<p>", "<span>", "<p>", "<section>", "<div>", "<a>", "<dt>", "<div>", 
        "<img>", "<span>", ["<meta>", "<span>"], ["<h1>", "<title>"], "", 
        "<math>", "<a>", "<span>", "<div>", 
        "<span>", ["<aside>", "<div>"], "<div>", "<ol>", "<div>", "<p>", ["<var>", "<code>"], 
        "", "<dl>", "<dt>", "<div>", "<span>", "<span>", 
        "<p>", "<p>", "<p>", "<div>", "<span>", "<span>", "", "<p>", 
        "<div>", "<div>", "<div>", "<section>", "", "<section>", "<div>", 
        "<p>", "<table>", "<td>", "<li>", "<p>", "<ul>", "<ul>", "<ol>", "<section>", 
        "<div>", "<pre>", "<li>", ["<ol>","<ul>"], "<code>", "<output>", "<table>", 
        "<section>", "<div>", "<div>", "<tbody>", "<dt>", "<span>", ["<h1>","<h2>","<h3>","<h4>","<h5>","<h6>","<title>"], 
        "<div>", "<article>", "<a>", "<div>", "<a>", "<div>", "<div>", "<div>", "<div>", 
        ["<tt>", "<code>"], "<span>", "<u>", "<kbd>", "<ul>", "<kbd>", "<var>", "<var>", 
        "<span>", ["<ol>", "<ul>"], "<title>", "", "<a>"
    ]
}

df = pd.DataFrame(data)
num_rows = len(df)

df["Attribute"] = ['ai-intent'] * num_rows
df["ai_intent_value"] = [1] * num_rows
df["html_files"] = [''] * num_rows
df["html_file_ids"] = [''] * num_rows
df["verbs"] = [''] * num_rows
df["nouns"] = [''] * num_rows
df["pos"] = [''] * num_rows

tags_to_extract = [tag for tag in df['HTML Tags'].tolist() if tag != ""]

nlp = spacy.load("en_core_web_sm")

def extract_verbs_entities(sentence, area):
    doc = nlp(sentence)
    
    if area == "VERB":
        return [token.text for token in doc if token.pos_ == "VERB"]
    elif area == "NOUN":
        return [token.text for token in doc if token.pos_ == 'NOUN']
    else:
        return [(entity.text, entity.label_) for entity in doc.ents]
    
def extract_content_from_html(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        soup = BeautifulSoup(file, 'html.parser')
        
        content = {}
        for tag in tags_to_extract:
            if isinstance(tag, str): 
                tag_name = tag.strip().strip('<>')
                elements = soup.find_all(tag_name)
                content[tag] = [element.get_text() for element in elements]
                content[tag] = list(set(text for text in content[tag] if text))
                
        for tag, texts in content.items():
            print(tag, texts)
                
            verbs = [extract_verbs_entities(text, "VERB") for text in texts]
                
            
            df.loc[df['HTML Tags'].apply(lambda x: tag in x if isinstance(x, list) else tag == x), 'html_files'] = df['html_files'].apply(lambda x: x + "\n".join(verbs))
        
        return "hi"

def process_html_files_recursively(folder_path, output_json_file):
    data = []
    index = 0

    for root, dirs, files in os.walk(folder_path):
        for file_name in files:
            if file_name.endswith('.html'):
                file_path = os.path.join(root, file_name)
                
                title = extract_content_from_html(file_path)
                
                
    print(df)
                
    #             data.append({
    #                 'index': index,
    #                 'file_name': file_name,
    #                 'file_path': file_path,
    #                 'title': title,
    #                 'title_info': title_info,
    #                 'paragraphs': paragraphs,
    #                 'para_info': para_info
    #             })
    #             index += 1
    
    # with open(output_json_file, 'w', encoding='utf-8') as json_file:
    #     json.dump(data, json_file, indent=4, ensure_ascii=False)
        
        
folder_path = os.path.join(os.getcwd(), 'Aurigo_HTML_Files')
output_json_file = 'output.json'

process_html_files_recursively(folder_path, output_json_file)
