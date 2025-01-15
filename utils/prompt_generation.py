import numpy as np
import pickle
import openai
import re
import pandas as pd

class PromptGeneration: 
    """
    Class for constructing text prompts based on tabular patient records
    df: dataframe containing a set of tabular patient records, generated using the RespiratoryData class
    """

    # probability of mentioning a symptom in the text, if the symptom is positive or negative
    prob_mention_symptom_if_positive = {"dysp": 0.95, "cough": 0.95, "pain": 0.75, "low_fever": 0.7, "high_fever": 0.95, "nasal": 0.95}
    prob_mention_symptom_if_negative = {"dysp": 0.75, "cough": 0.9, "pain": 0.3, "fever": 0.4, "nasal": 0.1}


    # symptom descriptors used to describe symptoms, conditional on what caused the symptom
    descriptors = {

        "dysp": {
            "asthma": ["attack-related", "at night", "in episodes", "wheezing", "difficulty breathing in", "feeling of suffocation", 
                    "nighttime stuffiness", "provoked by exercise", "light", "severe", "not able to breathe properly", "air hunger"], 
            "smoking": ["during exercise", "worse in morning", "mild"], 
            "COPD": ["chronic", "worse during flare-up", "worse while lying down", "difficulty sleeping", "air hunger"],
            "hay_fever": ["light", "mild", "stuffy feeling", "all closed up"], 
            "pneu": ["light", "mild", "severe", "no clear cause"]
            }, 
        
        "cough": {
            "asthma": ["attack-related", "dry"], 
            "smoking": ["productive", "mostly in morning", "during exercise", "gurgling"], 
            "COPD": ["phlegm", "sputum", "gurgling", "worse while lying down"], 
            "pneu": ["for over 7 days", "light", "mild", "severe", "non-productive at first, later purulent"], 
            "inf": ["prickly", "irritating", "dry", "phlegm", "sputum", "light", "mild", "severe", "constant", "day and night"]
        },
        
        "pain": {
            "pneu": ["light", "mild", "severe", "localized on right side", "localized on left side", "associated with breathing"], 
            "COPD": ["light", "mild"], 
            "inf": ["burning pain in trachea", "burning pain in windpipe", "scraping pain in trachea", "scraping pain in windpipe", 
                    "light", "mild"], 
            "cough": ["muscle pain", "burning pain in trachea", "burning pain in windpipe", "scraping pain in trachea", 
                    "scraping pain in windpipe"], 
            "asthma": ["tension behind sternum"]
        }
        
    }

    # strings used in the prompt, to describe symptoms
    symptom_strings = {"dysp": "dyspnea", "cough": "cough", "pain": "respiratory pain", "fever": "fever", "nasal": "nasal symptoms"}

    def __init__(self, df, seed): 
        
        self.df = df[["asthma", "smoking", "COPD", "hay_fever", "pneu", "inf", "dysp", "cough", "pain", "fever", "nasal"]]
        np.random.seed(seed)

    def symptom_mentioned(self, patient, symptom): 
        # returns whether symptom is mentioned or not, for patient (=row in df)
        
        if patient[symptom] == "no" or patient[symptom] == "none": 
            p = PromptGeneration.prob_mention_symptom_if_negative[symptom]
        elif symptom == "fever":
            cat = patient[symptom]+"_fever"
            p = PromptGeneration.prob_mention_symptom_if_positive[cat]
        else: 
            p = PromptGeneration.prob_mention_symptom_if_positive[symptom]

        mention = np.random.choice([True, False], 1, p=[p, 1-p])[0]
        return mention
    
    def symptom_descriptor(self, patient, symptom): 
        # returns descriptor describing symptom (randomly sampled based on cause), for patient (=row in df)
        
        causes = []
        for cause in PromptGeneration.descriptors[symptom].keys(): 
            if patient[cause] == "yes": 
                causes.append(cause)

        if patient[symptom] == "yes" and len(causes) != 0: # if the symptom is "on" and there are causes
            
            if "pneu" in causes: # strongest cause: pneu
                possible_descr = PromptGeneration.descriptors[symptom]["pneu"]
            elif "inf" in causes: # second-strongest cause: inf
                possible_descr = PromptGeneration.descriptors[symptom]["inf"]
            else: # if no stronger causes (pneu or inf), make bag of all descriptors 
                possible_descr = []
                for cause in causes: 
                    possible_descr += PromptGeneration.descriptors[symptom][cause]
                    
            descr = np.random.choice(possible_descr) # choose a random descriptor, uniformly
                
        else: # if symptom not present or has no particular causes, then no descriptors are sampled
            descr = ""
            
        return descr
    
    def symptoms_string(self, patient):
        # create part of prompt listing the symptoms to be mentioned, including their descriptors
        
        strings = []
        for symptom in ["dysp", "cough", "pain", "fever", "nasal"]:
            if patient[symptom+"_mention"]: # symptom is mentioned
                if patient[symptom] != "no" and patient[symptom] != "none": # symptom is turned on
                    string = f"- {PromptGeneration.symptom_strings[symptom]}: {patient[symptom]}"
                    if symptom != "fever" and symptom != "nasal":
                        descr = patient[symptom+"_descr"] # symptom descriptor
                        if len(descr) != 0: 
                            string += f", {descr}"
                    string += "\n"
                    strings.append(string)
                else: # symptom is not turned on
                    string = f"- {PromptGeneration.symptom_strings[symptom]}: {patient[symptom]}\n"
                    strings.append(string)

        np.random.shuffle(strings) # random reordering of symptoms
        string = "".join(strings) 
        
        return string
    
    def no_mention_symptoms_string(self, patient):
        # create part of prompt listing the symptoms to not be mentioned
    
        strings = []
        for symptom in ["dysp", "cough", "pain", "fever", "nasal"]:
            if not patient[symptom+"_mention"]: # symptom is not mentioned
                strings.append(f"- {PromptGeneration.symptom_strings[symptom]}\n")

        np.random.shuffle(strings) # random reordering of symptoms
        string = "".join(strings) 
        
        return string
    
    def background_conditions(self, patient): 
        # create part of prompt listing underlying respiratory conditions
    
        background = []
        bg_dict = {"asthma": "asthma", "COPD": "COPD", "smoking": "smoking", "hay_fever": "hay fever"}
        for bg in ["asthma", "COPD", "smoking", "hay_fever"]: 
            if patient[bg] == "yes": 
                background.append(f"- {bg_dict[bg]}\n")
                
        np.random.shuffle(background)
        bg_string = "".join(background)
        
        return bg_string
    
    def create_prompt_positive(self, patient): 
        # fill in base prompt with parts 1 (symptoms to mention), 2 (symptoms not to mention) and 3 (underlying respiratory conditions)

        base_prompt = """Create a short clinical note related to the following patient encounter. 

The following information is known about the patient's symptoms:
{symptoms_string}{no_mention_symptoms_string}{background_conditions_string}
The note has the following structure: 
**History**
<history>
**Physical Examination**
<physical examination results>

Do not include any suspicions of possible diagnoses in the clinical note (no "assessment" field). You can imagine additional context or details described by the patient, but no additional symptoms. Do not mention patient gender or age. Your notes can be relatively long (around 5 lines or more in history).

Do not add a title. Do not add a final comment after generating the note. """
    
        if len(patient["no_mention_symptoms_string"]) != 0: 
            no_mention_symptoms_string = "\nDon't mention anything about the following symptoms:\n"
            no_mention_symptoms_string += patient["no_mention_symptoms_string"]
        else: 
            no_mention_symptoms_string = ""
        
        if len(patient["background_string"]) != 0: 
            background_conditions_string = "\nThe patient currently has the following underlying health conditions, which may or may not be mentioned in the note if relevant:\n"
            background_conditions_string += patient["background_string"]
        else: 
            background_conditions_string = ""
            
        return base_prompt.format(symptoms_string=patient["symptoms_string"], no_mention_symptoms_string=no_mention_symptoms_string, background_conditions_string=background_conditions_string)
    

    def create_prompt_negative_bg(self, patient): 

        base_prompt = """Create a short clinical note related to the following patient encounter.
            
The patient does not experience any of the following symptoms:
- fever
- dyspnea
- chest pain / pain attributed to airways
- sneezing / blocked nose
- cough
            
The patient currently has the following underlying health conditions, which may or may not be mentioned in the note if relevant:
{background_conditions_string}
The note has the following structure: 
**History**
<history>
**Physical Examination**
<physical examination results>

Do not include any suspicions of possible diagnoses in the clinical note (no "assessment" field). You can imagine context or details described by the patient. Do not mention patient gender or age. Your notes can be relatively long (around 5 lines or more in history).

Do not add a title. Do not add a final comment after generating the note. """
        
        return base_prompt.format(background_conditions_string=patient["background_string"])
        
    def generate_prompts_positive(self): 

        # select all patients where at least one symptom is positive
        df_pos_symptoms = self.df.loc[(self.df["dysp"]!="no")|(self.df["cough"]!="no")|(self.df["pain"]!="no")|(self.df["fever"]!="none")|(self.df["nasal"]!="no")].copy()

        # add columns indicating whether each symptom should be mentioned in the text
        for symptom in ["dysp", "cough", "pain", "fever", "nasal"]:
            df_pos_symptoms[symptom+"_mention"] = df_pos_symptoms.apply(lambda row: self.symptom_mentioned(row, symptom), axis=1)

        # add columns containing symptom descriptors
        for symptom in ["dysp", "cough", "pain"]:
            df_pos_symptoms[symptom+"_descr"] = df_pos_symptoms.apply(lambda row: self.symptom_descriptor(row, symptom), axis=1)

        # first part of prompt: list symptoms that should be mentioned, including descriptors
        df_pos_symptoms["symptoms_string"] = df_pos_symptoms.apply(self.symptoms_string, axis=1) 

        # second part of prompt: list symptoms that should not be mentioned
        df_pos_symptoms["no_mention_symptoms_string"] = df_pos_symptoms.apply(self.no_mention_symptoms_string, axis=1)

        # third part of prompt: list underlying respiratory conditions
        df_pos_symptoms["background_string"] = df_pos_symptoms.apply(self.background_conditions, axis=1)
        
        # fill in prompt with first, second and third part
        df_pos_symptoms["prompt"] = df_pos_symptoms.apply(self.create_prompt_positive, axis=1)

        return df_pos_symptoms
    
    def generate_prompts_negative_bg(self): 

        # select all patients where none of the symptoms are positive
        df_neg_symptoms = self.df.loc[(self.df["dysp"]=="no")&(self.df["cough"]=="no")&(self.df["pain"]=="no")&(self.df["fever"]=="none")&(self.df["nasal"]=="no")].copy()

        # select all patients with at least one underlying respiratory condition
        df_neg_symptoms_bg = df_neg_symptoms.loc[(df_neg_symptoms["asthma"]=="yes")|(df_neg_symptoms["smoking"]=="yes")|(df_neg_symptoms["COPD"]=="yes")|(df_neg_symptoms["hay_fever"]=="yes")].copy()
        
        # second part of prompt: list underlying respiratory conditions
        df_neg_symptoms_bg["background_string"] = df_neg_symptoms_bg.apply(self.background_conditions, axis=1)

        # fill in prompt with second part only, use special prompt
        df_neg_symptoms_bg["prompt"] = df_neg_symptoms_bg.apply(self.create_prompt_negative_bg, axis=1)

        return df_neg_symptoms_bg

        
def prompt_GPT(patient): 
    """
    prompt GPT4-o for patient (row in dataframe, column "prompt" contains the prompt)
    """
    
    if (len(patient["prompt"]) != 0) and (len(patient["text_note"])==0): # prompt exists and we have not generated a note yet
    
        messages = []

        system_message = {"role": "system", "content": "You are a general practitioner, and need to summarize the patient encounter in a clinical note. Your notes are detailed and extensive. "}
        messages.append(system_message)

        messages.append({"role": "user", "content": patient["prompt"]})

        res = openai.chat.completions.create(
            model = "gpt-4o", 
            temperature = 1.2, 
            max_tokens = 1000,
            messages = messages, 
        )

        text_note = res.choices[0].message.content # response
        
    else: 
        text_note = ""

    return text_note

def prompt_GPT_no_symptom_no_bg(): 

    base_prompt = """Create 3 short clinical notes related to the following patient encounter.
 
    The patient does not experience any of the following symptoms:
    - fever
    - dyspnea
    - chest pain / pain attributed to airways
    - sneezing / blocked nose
    - cough
    
    The patient does not have any of the following health conditions, so don't mention these: 
    - asthma
    - COPD
    - smoking
    - hay fever
    
    The note has the following structure: 
    **History**
    <history>
    **Physical Examination**
    <physical examination results>
    
    Do not include any suspicions of possible diagnoses in the clinical note (no "assessment" field). You can imagine context or details described by the patient. Do not mention patient gender or age. Your notes can be relatively long (around 5 lines or more in history).

    Separate the individual notes with "---". Do not add a title. Do not add a final comment after generating the note. """

    messages = []

    system_message = {"role": "system", "content": "You are a general practitioner, and need to summarize the patient encounter in a clinical note. Your notes are detailed and extensive. "}
    messages.append(system_message)

    messages.append({"role": "user", "content": base_prompt})

    res = openai.chat.completions.create(
        model = "gpt-4o", 
        temperature = 1.2, 
        max_tokens = 1500,
        messages = messages, 
    )

    text_note = res.choices[0].message.content # response
    
    return text_note

def split_unrelated_notes(response): 
    split_note = re.split('-{2,}', response)
    if len(split_note[0]) == 0: 
        split_note = split_note[1:]
    if len(split_note[-1]) == 0:
        split_note = split_note[:-1]
    if split_note[-1] == "\n" or split_note[-1] == "\n\n":
        split_note = split_note[:-1]
    if len(split_note) == 3: 
        return split_note
    else: 
        print("WARNING! response cannot be split into separate notes")
        return []
    
def prompt_GPT_advanced_note(patient): 
    
    if (len(patient["prompt"]) != 0) and (len(patient["advanced_note"])==0): # prompt exists and there we have not generated a note yet
    
        messages = []

        system_message = {"role": "system", "content": "You are a general practitioner, and need to summarize the patient encounter in a clinical note. Your notes are detailed and extensive. "}
        messages.append(system_message)
        messages.append({"role": "user", "content": patient["prompt"]}) # prompt used to get original text note
    
        messages.append({"role": "assistant", "content": patient["text_note"]}) # add the note as the previous reply
        messages.append({"role": "user", "content": "Please write this note in more compact style (using abbreviations and shortcuts), while preserving the content."})

        res = openai.chat.completions.create(
            model = "gpt-4o", 
            temperature = 1.2, 
            max_tokens = 1000,
            messages = messages, # with assistant response and additional telegram-style request added
        )

        text_note_shorter = res.choices[0].message.content

    return text_note_shorter
