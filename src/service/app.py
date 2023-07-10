from fairseq.models.transformer import TransformerModel
from flask import render_template, request, jsonify, Flask
import logging
import json
from interaction_org_helper import InteractionHelper
from mosestokenizer import *
import traceback
import flask

from flask.wrappers import Response

from translation_connector import TranslationConnector

app = Flask(__name__)
app.config['JSON_AS_ASCII'] = False
PORT = TranslationConnector.PORT+1
HOST = TranslationConnector.HOST

trained_model = dict()
logger = logging.getLogger("main")



@app.route('/', methods=['GET', 'POST'])
def translate_home():
    if request.method == "GET":
        return render_template('greeting.html', question='Hello world!', translation_ids=list(trained_model))
    question = request.form['question']
    translation_id = request.form['translation_id']
    logger.info(question)
    logger.info(translation_id)

    translation_info = dec_model_info(translation_id)
    src_sent = question
    src_lang = translation_info.get("srclang")
    tgt_lang = translation_info.get("tgtlang")
    model_id = translation_info.get("modelid", "transformer")
    out = __translate(src_sent, src_lang, tgt_lang, model_id, None)
    
    return render_template('greeting.html', question=question, answer=out.get("result"), 
        detail_ans=json.dumps(out, indent=2, ensure_ascii=False), translation_ids=list(trained_model), translation_id_cur=translation_id)

def dec_model_info(id_str):
    fields = id_str.split(":")
    info = {}
    for f in fields:
        ff = f.split("=")
        if len(ff) > 1:
            info[ff[0]] = ff[1]
    return info

def jsonstr_return(data):
    json_response = json.dumps(data, ensure_ascii = False)
    #creating a Response object to set the content type and the encoding
    response = Response(json_response, content_type="application/json; charset=utf-8" )
    return response

@app.route('/list-models', methods=['GET'])
def list_models():
    out = {'result': list(trained_model)}
    logger.info(out)
    return jsonify(out)


def __translate(src_sent, src_lang, tgt_lang, model_id, src_sent_s=None):

    try:
        translation_info = enc_model_info(src_lang=src_lang, tgt_lang=tgt_lang, model_id=model_id)
        if translation_info in trained_model:
            # try:
            #     tokenize = MosesTokenizer(src_lang)
            #     src_sent = " ".join(tokenize(src_sent))
            # except Exception as e:
            #     pass

            tgt_sent = trained_model[translation_info].translate([src_sent] if src_sent is not None else src_sent_s)
            optional_info = None
            if isinstance(tgt_sent, tuple):
                tgt_sent, optional_info = tgt_sent[0], tgt_sent[1]
            out = {'result': tgt_sent,
                    'optional_info': optional_info,
                    }
        else:
            out = {'result': None,
                    'optional_info': 'model "{}" is not found'.format(translation_info),
                    }
    except Exception as e:
        traceback.print_stack()
        print(e)
        print("ERR when loading model: {}".format(m))
        print(e)
        out = {'result': None,
               'optional_info': str(e),
               }
    out['translation_info'] = translation_info
    logger.info(out)
    return out


@app.route('/translate-debug', methods=['GET', 'POST'])
def translate_debug():
    src_sent = request.args.get("srcsent")
    src_lang = request.args.get("srclang")
    tgt_lang = request.args.get("tgtlang")
    model_id = request.args.get("modelid", "transformer")
    out = __translate(src_sent, src_lang, tgt_lang, model_id)
    return jsonstr_return(out)


@app.route('/translate', methods=['POST'])
def translate():
    inputs = request.get_json(force=True)
    logger.info(inputs)
    src_sent = inputs.get("srcsent")
    src_lang = inputs.get("srclang")
    tgt_lang = inputs.get("tgtlang")
    translation_model_id = inputs.get("modelid", "transformer")

    out = __translate(src_sent, src_lang, tgt_lang, translation_model_id)

    return jsonstr_return(out)


def enc_model_info(**kwargs):
    id_str = ""
    for k in sorted(kwargs):
        v = kwargs[k]
        id_str = id_str + (":{}={}" if len(id_str) > 0 else "{}={}").format(k.replace("_", ""),v) 
    return id_str

def __reset_all_models(path_config_file='modelinfo.json'):
    global trained_model 
    trained_model = {}
    models_info = json.load(open(path_config_file, encoding="utf8"))
    for m in models_info:
        src_lang = m['translation_info']['srclang']
        tgt_lang = m['translation_info']['tgtlang']
        model_id = m['translation_info']['modelid']
        modelid = enc_model_info(src_lang=src_lang, tgt_lang=tgt_lang, model_id=model_id)
        print("Loading model [{}] ...".format(modelid))
        try: 
            data_path = m['model_info']['data_path']
            m['model_info'].pop('data_path')

            input_args = [data_path]
            for k, v in m['model_info'].items():
                input_args.append("--{}".format(k))
                input_args.append("{}".format(v))

            translation_helper = InteractionHelper(input_args=input_args)
            trained_model[modelid] = translation_helper

        except Exception as e:
            traceback.print_stack()
            print(e)
            print("ERR when loading model: {}".format(m))
    out = {'result': list(trained_model)}
    return out


@app.route('/reload-all-model', methods=['GET'])
def reset_all_model():
    return jsonify(__reset_all_models())
 
def _process(src_lang, tgt_lang):
    """Example method. Receive and respond with plaintext sentences."""
    data = str(flask.request.get_data(), encoding="utf-8")
    # print("Received: ", data)
    lines = [l.strip() for l in (data.split("\n") if "\n" in data else [data])]
    
    # scramble
    translated_sentences = __translate(None, src_lang, tgt_lang, 'phrase-transformer', src_sent_s=lines)['result']

    scrambled = "\n".join(translated_sentences)
    # print("Response: ", scrambled)
    return flask.make_response(scrambled)


@app.route("/test", methods=["POST"])
def test():
    return _process('zh', 'vi')

@app.route("/test-vizh", methods=["POST"])
def test_vizh():
    return _process('vi', 'zh')


if __name__ == "__main__":
    print(__reset_all_models(path_config_file='src/service/modelinfo_vlsp.json'))
    app.run("0.0.0.0", port=7080)
