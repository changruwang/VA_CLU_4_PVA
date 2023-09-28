import os
from azure.core.credentials import AzureKeyCredential
from azure.ai.language.conversations.authoring import ConversationAuthoringClient
from azure.core.rest import HttpRequest
from dotenv import load_dotenv
from datetime import date
import json
import pandas as pd
import spacy
import glob, pathlib
import yaml

load_dotenv()

class CustomNER:

    def __init__(self):
        # self.model = spacy.load(r'./entity-detection-model/') #TODO: custom spacy model will be replaced with custom NER in Cognitive Services
        self.model = spacy.load(r'entity-detection-model/spancat/') #TODO: custom spacy model will be replaced with custom NER in Cognitive Services
        self.dataset_path = '../../data/CLU/*/*.csv'
    

    def parse_output(self, utterance, intent, dataset='Train'):
        doc    = self.model(utterance)
        result = {'language': 'en-us',
                    'text': str(utterance).lower(),
                    'intent': intent,
                    'dataset': dataset
                }
        entity = [[e.start_char, e.text, e.label_] for e in doc.spans['sc']]
        entList = []
        for ent in entity:
            tmp = {}
            tmp['category'] = ent[2]
            tmp['category'] = tmp['category'].lstrip('From') ##TODO: remove in future
            tmp['category'] = tmp['category'].lstrip('To') ##TODO: remove in future
            tmp['offset']   = ent[0]
            tmp['length']   = int(len(ent[1]))
            
            entList.append(tmp)

        result['entities'] = entList

        return result
    
    def label_utterances(self):
        merged = []
        files = glob.glob(self.dataset_path)
        print(files)
        for file in files:
            #load csv and drop duplicates
            print(f'Reading file: {file}')
            df = pd.read_csv(file, sep='\t', header=None)
            df.drop_duplicates(inplace=True) #TODO: debug with dup data
            data   = df.iloc[:, 0].values
            tmp    = pathlib.PurePath(file)

            #detect intent name from folder
            intent  = tmp.parent.name

            #detect train/test set from the path
            dataset = str(tmp.name).strip('.csv')
            dataset = dataset.title()

            output = list(map(lambda utterance: self.parse_output(utterance, intent, dataset), data))
            merged.extend(output)

        return merged 
    
    def get_prebuilt_components(self, prebuilt_list: list) -> list:
        prebuilts = []
        for pb in prebuilt_list:
            prebuilts.append({'category': str(pb)})

        return prebuilts
    
    def format_category(self, *category) -> dict:
        return [{'category': cat} for cat in category]
                 
    
    def format_intents(self, intents: list) -> list:
        output = []
        for intent in intents:
            output.append({"category": str(intent)})

        return output
    
    def format_entity_config(self, entity_details: dict) -> dict:
        output = {}
        output['category']           = entity_details['name']
        output['compositionSetting'] = entity_details['compositionSetting']

        if 'list' in entity_details:
            tmp_list = []
            for list_object in entity_details['list']:
                #get key and synonyms
                key = list_object['key']
                val = list_object['synonyms']
                tmp_list.append({"listKey": key, "synonyms": [{"language": "en-us", "values": val}]})

            sublist = {'sublists': tmp_list}
            output['list'] = sublist
             

        if 'prebuilts' in entity_details:
            output['prebuilts']          = self.get_prebuilt_components(entity_details['prebuilts'])

        if 'regex' in entity_details:
            regex = {}
            tmp   = [{'regexKey': key, 'language': 'en-us', 'regexPattern': str(val)} for tmp in entity_details['regex'] for key, val in tmp.items() ]
            regex['expressions'] = tmp
            output['regex']      = regex

        if 'requiredComponents' in entity_details:
            output['requiredComponents'] =  entity_details['requiredComponents']


        return output




class CLU:

    def __init__(self):
        self.endpoint   = os.getenv('AZURE_CONVERSATIONS_ENDPOINT')
        self.key        = os.getenv('AZURE_CONVERSATIONS_KEY')
        self.project    = os.getenv('AZURE_CONVERSATIONS_CLU_PROJECT_NAME')
        self.deployment = os.getenv('AZURE_CONVERSATIONS_DEPLOYMENT_NAME')
        
        credential  = AzureKeyCredential(self.key)
        self.client = ConversationAuthoringClient(self.endpoint, credential)
        self.training_mode = 'standard'
        

    def create_project(self, intent_list: list, entity_list: list, utterance_list: list) -> str:
        #setup CLU metadata
        project_assets = {
            'projectKind': 'Conversation',
            'intents': intent_list,
            'entities': entity_list,
            'utterances': utterance_list
        }
        
            
        project={
            'projectFileVersion': '2023-04-01',
            'stringIndexType': 'Utf16CodeUnit',
            'metadata': {
                'projectKind': 'Conversation',
                'settings': {'confidenceThreshold': 0.3},
                'projectName': self.project,
                'multilingual': True,
                'description': 'Veteran Affairs Conversational Language Understanding Project',
                'language': 'en-us',
            },
            'assets': project_assets,
            
        }

        self.import_project(self.project, project) 
    

    def import_project(self, new_project_name, clu_project) -> str:

        print(f"Importing project as '{self.project}'")
        poller   = self.client.begin_import_project(project_name=new_project_name, project=clu_project)
        response = poller.result()
        print(f"Import project status: {response['status']}")

        return response['status']


    def export_project(self) -> bool:

        poller    = self.client.begin_export_project(project_name=self.project, string_index_type='Utf16CodeUnit', 
                                                  exported_project_format='Conversation')
        job_state = poller.result()

        print(f"Export project status: {job_state['status']}")
        request  = HttpRequest('GET', job_state['resultUrl'])
        response = self.client.send_request(request)
        exported = response.json()
        archived = 'archive/' + self.project + '_' + str(date.today()) + '.json'

        #save the exported file
        with open(archived, 'w') as fh:
            json.dump(exported, fh)

        #validate if the file was created
        if os.path.exists(archived) and os.path.getsize(archived) > 0:
            return True 
        
        return False
    

    def delete_project(self) -> None:

        poller = self.client.begin_delete_project(project_name=self.project)
        poller.result()
        print(f"Deleted project {self.project}")
    

    def train_model(self) -> str:
        poller = self.client.begin_train(project_name=self.project, 
                                         configuration={'modelLabel': 'Standard', 'trainingMode': self.training_mode,
                                                        'evaluationOptions': {
                                                            'kind': 'percentage',
                                                            'testingSplitPercentage':20,
                                                            'trainingSplitPercentage':80}
                                                        },
                                        )

        response = poller.result()
        print(f"Train model status: {response['status']}")

        return response['status']
    

    def deploy_model(self):
        print('Deploying model...')
        poller = self.client.begin_deploy_project(project_name=self.project, 
                                                  deployment_name=self.deployment,
                                                  deployment={'trainedModelLabel': 'Standard'},)
        response = poller.result()
        print(f"Model '{response['modelId']}' deployed to '{response['deploymentName']}'")

        return response['modelId']


def main():
    # #Create a new CLU Project using the config settings
    clu = CLU()
    ner = CustomNER()

    # # NOTE line 236 - 241 will archive an old clu model and delete it from language studio
    # # Export the CLU Project and archive it locally.
    # exported = clu.export_project() #TODO: additional validation needed to check if the model exists
    # print('exported: ', exported) #TODO: if export was successful then delete
    # if exported:
    #     #remove the project from Language Studio
    #     clu.delete_project()

    #lines 244 - 282 will create a new clu project, train it and create a deployment.
    with open('./config/settings.yml', 'r') as fh:
        clu_config    = yaml.safe_load(fh)

        entity_config = []
        # #format intents
        intent_config = ner.format_intents(clu_config['intents'])
        print(intent_config)

        # format entities
        # #amount
        amount_config = ner.format_entity_config(clu_config['entities']['Amount'])
        entity_config.append(amount_config)

        #date
        date_config = ner.format_entity_config(clu_config['entities']['datetimeV2'])
        entity_config.append(date_config)

        #Number
        number_config = ner.format_entity_config(clu_config['entities']['Number'])
        entity_config.append(number_config)

        #AccountType
        account_config = ner.format_entity_config(clu_config['entities']['AccountType'])
        entity_config.append(account_config)

        #TransactionType
        trans_config = ner.format_entity_config(clu_config['entities']['TransactionType'])
        entity_config.append(trans_config)

        #BenefitType
        benefit_config = ner.format_entity_config(clu_config['entities']['BenefitType'])
        entity_config.append(benefit_config)
        print(entity_config)

        #load utterances and label them using the custom spacy model
        utterances = ner.label_utterances()

        #create the new clu project
        clu.create_project(intent_config, entity_config, utterances)

    #Train & Deploy the CLU Model : #TODO: additional validation
    train_status = clu.train_model()
    if train_status == 'succeeded':
        clu.deploy_model()



if __name__ == '__main__':
    main()
