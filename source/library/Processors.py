import json
from library.BPMNGraph import BPMNGraph


class AgendapuntenProcessor:
    def __init__(self, llm_model, taxonomy_path = None):
        self.llm_model = llm_model

        if taxonomy_path is not None:
            self.taxonomy = json.load(open(taxonomy_path))

    def extract_keywords(self, agendapunt):
        # Convert the JSON object to a string
        text = json.dumps(agendapunt)

        # Generate the classification task
        prompt_string, system_message, task_string, context = self.llm_model.generate_keyword_task(text)

        # Generate the response using OpenAI (or any other method)
        response = self.llm_model.generate_response(system_message, prompt_string, stream=False, json_mode=True)
        response_json = self.llm_model.extract_json(response)

        if response_json is not None:
            task_info = self.llm_model.get_task_info("keywords_agendapunt", "keywords", "keywords_agendapunt", "json", "json")
            saveable = self.llm_model.formatted_results(task_info, text, task_string, context, system_message, prompt_string, response)

        return response_json, saveable

    def translate(self, agendapunt, language, agenda_punten_format = None):

        if agenda_punten_format is None:
            agenda_punten_format = """{'id': 'Do not translate', 'name': 'Translated Name', 'source': 'Source language', 'target': 'Target language'}"""

        prompt_string, system_message, task_string, context = self.llm_model.generate_translation_task(agendapunt, language, agenda_punten_format)
        response = self.llm_model.generate_response(system_message,prompt_string)

        response_json = self.llm_model.extract_json(response)

        if response_json is not None:
            task_info = self.llm_model.get_task_info("translate_agendapunt", "translate", "translate_agendapunt", "json", "json", language)
            saveable = self.llm_model.formatted_results(task_info, None, task_string, context, system_message, prompt_string, response)

        return response_json, saveable
    
    def classify(self, agendapunt, taxonomy = None):
    
        if taxonomy is None:
            taxonomy = self.taxonomy
            
        # Convert the JSON object to a string
        text = json.dumps(agendapunt)

        # Generate the classification task
        prompt_string, system_message, task_string, context = self.llm_model.generate_classification_task(text, taxonomy)

        # Generate the response using OpenAI (or any other method)
        response = self.llm_model.generate_response(system_message, prompt_string, stream=False, json_mode=False)
        response_json = self.llm_model.extract_json(response)

        if response_json is not None:
            task_info = self.llm_model.get_task_info("classification_agendapunt", "classification", "classification_agendapunt", "json", "json") 
            saveable = self.llm_model.formatted_results(task_info, text, task_string, context, system_message, prompt_string, response)

        return response_json, saveable
    
class BPMNProcessor:

    def __init__(self, llm_model, taxonomy_path = None):
        self.llm_model = llm_model

        if taxonomy_path is not None:
            self.taxonomy = json.load(open(taxonomy_path))

    def get_data_bpmn(self, graph, with_documentation=True):
        translation_jobs = []
        for id, node in graph.get_nodes():
            node_name = node["name"]
            
            if with_documentation:
                node_documentation = node["documentation"]
                if node_name != '' or node_documentation != '':
                    translation_jobs.append({
                        'id': id,
                        'name': node_name,
                        'documentation': node_documentation
                    })
            else:
                if node_name != '':
                    translation_jobs.append({
                        'id': id,
                        'name': node_name
                    })

        return translation_jobs

    def extract_keywords(self, file_path):
        # Create a BPMN graph
        graph = BPMNGraph(data = file_path)

        classification_data = self.get_data_bpmn(graph)
        classification_text = json.dumps(classification_data)

        prompt_string, system_message, task_string, context = self.llm_model.generate_keyword_task(classification_text)
        response = self.llm_model.generate_response(system_message,prompt_string, stream=False, json_mode=True)
        response_json = self.llm_model.extract_json(response)
        if response_json is not None:
            task_info = self.llm_model.get_task_info("keywords_bpmn", "keywords", "keywords_bpmn", "bpmn", "json")
            saveable = self.llm_model.formatted_results(task_info, file_path, task_string, context, system_message, prompt_string, response)
        return response_json, saveable

    def translate(self, file_path, language, translation_format = None):
        # Create a BPMN graph
        graph = BPMNGraph(data = file_path)

        translation_jobs = self.get_data_bpmn(graph, with_documentation=False)

        if translation_format is None:
            translation_format = """{'id': 'Do not translate', 'name': 'Translated Name', 'documentation': 'Translated Documentation', 'source': 'Source language', 'target': 'Target language'}"""

        prompt_string, system_message, task_string, context = self.llm_model.generate_translation_tasks(translation_jobs, language, translation_format)
        response = self.llm_model.generate_response(system_message,prompt_string, stream=False, json_mode=False)
        response_json = self.llm_model.extract_json(response)  
        if response_json is not None:
            task_info = self.llm_model.get_task_info("translate_bpmn", "translate", "translate_bpmn", "bpmn", "json", language)
            saveable = self.llm_model.formatted_results(task_info, file_path, task_string, context, system_message, prompt_string, response)

        return response_json, saveable
    
    def classify(self, file_path, taxonomy = None):

        if taxonomy is None:
            taxonomy = self.taxonomy

        # Create a BPMN graph
        graph = BPMNGraph(data = file_path)

        classification_data = self.get_data_bpmn(graph)

        prompt_string, system_message, task_string, context = self.llm_model.generate_classification_task(classification_data, taxonomy)
        response = self.llm_model.generate_response(system_message,prompt_string, stream=False, json_mode=False)
        response_json = self.llm_model.extract_json(response)
        if response_json is not None:
            task_info = self.llm_model.get_task_info("classification_bpmn", "classification", "classification_bpmn", "bpmn", "json")
            saveable = self.llm_model.formatted_results(task_info, file_path, task_string, context, system_message, prompt_string, response)
        return response_json, saveable