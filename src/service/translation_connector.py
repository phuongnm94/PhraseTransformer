import requests
import json


class TranslationConnector:
    HOST, PORT = '0.0.0.0', 8008

    def __init__(self, port=None, host=None) -> None:
        self.host = host or self.HOST
        self.port = port or self.PORT

    def get_translation(self, s_in, source_lang, target_lang, model_id, template=None):
        headers = {'Content-type': 'application/json'}
        data = json.dumps({"srclang": source_lang, "tgtlang": target_lang,
                           "srcsent": s_in, "modelid": model_id, "template": template})

        response = requests.post('http://{}:{}/translate'.format(self.host, self.port),
                                 headers=headers, data=data)
        return response.json()['result']
