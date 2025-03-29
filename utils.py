import torch
from transformers import AutoModelForCausalLM

def process_prompt_stable(prompt, task_type, num_fewshot):
    if task_type == 'QA':
        pre_prompt = generate_few_shot_QA(num_fewshot)
        final_prompt = pre_prompt + prompt + '\nA:'
        
    elif task_type == 'Summarize':
        # pre_prompt = 'Summarize the following document: '
        pre_prompt = generate_few_shot_summarize(num_fewshot)
        final_prompt = pre_prompt + prompt
        # final_prompt = prompt
        
    elif task_type == 'translate_de_en':
        pre_prompt = generate_few_shot_translate_de_en(num_fewshot)
        final_prompt = pre_prompt + prompt + '\nEnglish phrase:'

    else:
        print("Task_type must be one of QA, Summarize and translate_de_en")
    
    return final_prompt

    



def generate_few_shot_QA(num_fewshot):
    # All available few-shot examples
    examples = [
        "Q: What is human life expectancy in the United States?\nA: Human life expectancy in the United States is 78 years.",
        "Q: Who was president of the United States in 1955?\nA: Dwight D. Eisenhower was president of the United States in 1955.",
        "Q: Which party did he belong to?\nA: He belonged to the Republican Party.",
        "Q: What is the square root of banana?\nA: I have no comment.",
        "Q: How does a telescope work?\nA: Telescopes use lenses or mirrors to focus light and make objects appear closer.",
        "Q: Where were the 1992 Olympics held?\nA: The 1992 Olympics were held in Barcelona, Spain."
    ]
    
    if num_fewshot < 1 or num_fewshot > len(examples):
        raise ValueError("Number of few-shot examples must be between 1 and 6.")
    
    few_shot_prompt = '\n\n'.join(examples[:num_fewshot])
    pre_prompt = few_shot_prompt + '\n\nQ: '
    
    return pre_prompt





def generate_few_shot_translate_de_en(num_fewshot):
    # All available few-shot examples
    examples = [
    "German phrase: Seiner Ansicht nach könnten alle Mitglieder beider Vereine künftig wieder an einem Strang ziehen.\nEnglish phrase: In his opinion, all members of each club could come together again in the future.",
    "German phrase: In Bad Salzhausen wurden die Sportler von Bürgermeister Hans-Peter Seum, der in Bad Vilbel mit dem Fahrrad mitgestartet war, und der Leiterin des Eigenbetriebs Bad Salzhausen, Petra Schwing-Döring, begrüßt.\nEnglish phrase: In Bad Salzhausen, the sportsmen-and-women were greeted by mayor Hans-Peter Seum, who had ridden off the start-line in Bad Vilbel on his own bike, and the boss of the municipal company Bad Salzhausen, Petra Schwing-Döring.",
    "German phrase: Als die Vereinigten Staaten nach dem Zweiten Weltkrieg Japan besetzte, ermutigten General Douglas Mac Arthur und sein Stab das Land dazu, eine Verfassung zu verabschieden, die sicherstellen solle, dass die militarisierte Autokratie Hedki Tojos durch Demokratie ersetzt würde.\nEnglish phrase: When the United States occupied Japan after World War II, General Douglas MacArthur and his aides encouraged the country to adopt a constitution designed to assure that Hideki Tojo's militarized autocracy would be replaced with democracy.",
    "German phrase: Henry führte 20 Zeilen aus Othellos letzter Rede für den Dokumentarfilm auf und er war begeistert.\nEnglish phrase: Henry delivered 20 lines of Othello's last speech for the documentary and he was hooked.",
    "German phrase: Für Adelaide ist am Dienstag eine Höchsttemperatur von 16°C vorhergesagt, mit möglichen Schauern.\nEnglish phrase: A top of 16C is forecast for Adelaide on Tuesday, with the chance of a shower or two.",
    "German phrase: Und es ist anstrengend.\nEnglish phrase: And it's tedious."]
    
    if num_fewshot < 1 or num_fewshot > len(examples):
        raise ValueError("Number of few-shot examples must be between 1 and 6.")
    
    few_shot_prompt = '\n\n'.join(examples[:num_fewshot])
    pre_prompt = few_shot_prompt + 'German phrase: '
    
    return pre_prompt






def generate_few_shot_summarize(num_fewshot):
    # All available few-shot examples
    examples = [
    "Text: The full cost of damage in Newton Stewart, one of the areas worst affected, is still being assessed. Repair work is ongoing in Hawick and many roads in Peeblesshire remain badly affected by standing water. Trains on the west coast mainline face disruption due to damage at the Lamington Viaduct. Many businesses and householders were affected by flooding in Newton Stewart after the River Cree overflowed into the town. First Minister Nicola Sturgeon visited the area to inspect the damage. The waters breached a retaining wall, flooding many commercial properties on Victoria Street - the main shopping thoroughfare. Jeanette Tate, who owns the Cinnamon Cafe which was badly affected, said she could not fault the multi-agency response once the flood hit. However, she said more preventative work could have been carried out to ensure the retaining wall did not fail. 'It is difficult but I do think there is so much publicity for Dumfries and the Nith - and I totally appreciate that - but it is almost like we're neglected or forgotten,' she said. 'That may not be true but it is perhaps my perspective over the last few days. 'Why were you not ready to help us a bit more when the warning and the alarm alerts had gone out?' Meanwhile, a flood alert remains in place across the Borders because of the constant rain. Peebles was badly hit by problems, sparking calls to introduce more defences in the area. Scottish Borders Council has put a list on its website of the roads worst affected and drivers have been urged not to ignore closure signs. The Labour Party's deputy Scottish leader Alex Rowley was in Hawick on Monday to see the situation first hand. He said it was important to get the flood protection plan right but backed calls to speed up the process. 'I was quite taken aback by the amount of damage that has been done,' he said. 'Obviously it is heart-breaking for people who have been forced out of their homes and the impact on businesses.' He said it was important that 'immediate steps' were taken to protect the areas most vulnerable and a clear timetable put in place for flood prevention plans. Have you been affected by flooding in Dumfries and Galloway or the Borders? Tell us about your experience of the situation and how it was handled. Email us on selkirk.news@bbc.co.uk or dumfries@bbc.co.uk.\nSummary: Clean-up operations are continuing across the Scottish Borders and Dumfries and Galloway after flooding caused by Storm Frank.",
    
    "Text: he announcement ends months of uncertainty for Cornish Language Partnership staff whose contracts had been due to end. Local government minister Andrew Stunnell said the three-year funding package for the service would help make sure the language survived. But he warned that long term funding should come from Cornwall. He said it was 'important to make sure the Cornish were given the opportunity to put down sound foundations.' 'In the longer term support for the Cornish language is going to be something which is going to have to be based in Cornwall and will not come from London,' he added. The Cornish Language Partnership's, Jennifer Lowe, said: 'We can now plan for the future thanks to the funding.' The United Nations recently upgraded the status of the Cornish language from 'extinct' to 'critically endangered'. It is thought fewer than 500 people worldwide are fluent in the language.\nSummary: The government is spending nearly £400,000 to help save the Cornish language.",

    "Text: Operation Equinox is investigating claims of sexual, physical and emotional abuse between the 1940s and 1990s. In a letter to victims Nottinghamshire Police confirmed 530 of 636 reported crimes were on council property. Officers also said 485 alleged offences were committed by council staff and of 432 suspects, 283 had been identified. More on this story and other news in Nottinghamshire So far, police have had 290 people report crimes. Operation Equinox combined two police inquiries. Operation Daybreak, sent up in 2011, was focussed on the Beechwood children's home in Nottingham, while Operation Xeres has been looking at residential homes in the county. The letter emphasises the progress already made, with former social worker Andris Logins jailed for 20 years. Two other men have been jailed for historical attacks not connected to children's homes and three more trials are due to begin in early 2017. Nottinghamshire Police has not commented directly as the information is part of an ongoing enquiry.\nSummary: An inquiry into historical abuse in Nottinghamshire has recorded more than 500 offences on council property."]
    
    if num_fewshot < 1 or num_fewshot > len(examples):
        raise ValueError("Number of few-shot examples must be between 1 and 3.")
    
    few_shot_prompt = '\n\n'.join(examples[:num_fewshot])
    pre_prompt = few_shot_prompt + '\nSummary: '
    
    return pre_prompt











def process_prompt_similarity(prompt, task_type):
    if task_type == 'QA':
        final_prompt = 'Question:' + prompt + 'Answer:'
        
    elif task_type == 'Summarize':
        pre_prompt = 'Summarize the following document: '
        final_prompt = pre_prompt + prompt
        
    elif task_type == 'translate_de_en':
        final_prompt = 'German phrase:' + prompt + '\nEnglish phrase:'

    else:
        print("Task_type must be one of QA and translate_de_en")
    
    return final_prompt



def process_data(dataset, dataset_name):
    
    if dataset_name == "truthful_qa":
        question = dataset['validation']['question']
        best_answer = dataset['validation']['best_answer']
        process_data = ['Question:' + a + 'Answer:' + b for a, b in zip(question, best_answer)]
        
    elif dataset_name == "wmt16-de-en":
        de = dataset['validation']['de']
        en = dataset['validation']['en']
        process_data = ['German phrase: ' + a + "\nEnglish phrase:" + b for a, b in zip(de, en)]

    return process_data
    

def load_opt_param(name, param, start_num, end_num, cpu_only: bool):
    num = 0
    if len(name.split('.'))>4:
        num = int(name.split('.')[3])                    
    if not (num>start_num and num<end_num and ('fc1' in name or 'fc2' in name)):
        if (cpu_only):
            param.data = param.data.to('cpu').half()
        else:
            param.data = param.data.to('cuda:0').half()

def load_llama_param(num, name, param, start_num, end_num, cpu_only: bool):
    num = 0
    if len(name.split('.'))>3:
        num = int(name.split('.')[2])
    if not (num>start_num and num<end_num and ('gate_proj' in name or 'up_proj' in name or 'down_proj' in name)):
        if (cpu_only):
            param.data = param.data.to('cpu').half()
        else:
            param.data = param.data.to('cuda:0').half()

def load_model_memory_limit(checkpoint_path, start_num, end_num, model_name, cpu_only: bool):
    model = AutoModelForCausalLM.from_pretrained(checkpoint_path, torch_dtype=torch.float16, device_map='cpu')
    num_layers = model.config.num_hidden_layers
    for name, param in model.named_parameters():
        if ("opt" in model_name):
            load_opt_param(name, param, start_num, end_num, cpu_only)
        elif ("llama" in model_name):
            load_llama_param(name, param, start_num, end_num, cpu_only)
            
    return model, num_layers