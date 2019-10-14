
from flask import Flask, request, make_response
from flask import jsonify
from flask_cors import CORS
import multiprocessing
import dataset
import sys, time, os
import uuid, json
from math import log

DATABASE_URL = 'sqlite:///samples.db'
M = 1000

def _get_file_endswith(folder,file_end):
    for file in os.listdir(folder):
        if file.endswith(file_end):
            return file

    return None

def _fetch_models(elem):
    path = elem.get("products")
    nb_elements = elem.get("nb_initial_config")
    json_path = "{}/{}products.json".format(path, nb_elements)

    def format_element(elem):
        structure = elem[1][1]
        structure = {"nb_blocks":structure[0], "nb_layers":structure[1], "nb_params":(structure[2]), "nb_flops":structure[3]}
        #structure = {"nb_layers":structure[1], "nb_params":structure[2]}
        
        robustness = elem[1][2][0]
        return {"accuracy":elem[0], "name":elem[1][0], **structure, "robustness":robustness}

    if os.path.isfile(json_path):
        
        
        with open(json_path) as file:
            obj = json.load(file)
            print(len(obj))
            return list(map(format_element,obj))
    
    return None
def _fetch_elements(elem, count_only=True):
    path = elem.get("products")
    elements = []
    for file in os.listdir(path):
        if file.endswith(".h5"):
            elements.append(file)
    if count_only:
        elem["nb_valid_elements"] = len(elements);
    else:
        elem["valid_elements"] = elements;
    return elem

def init_task(params, status="init"):
    db = dataset.connect(DATABASE_URL)

    id = str(uuid.uuid1())[:10]
    params["task_name"] = params.get("task_name","").replace(" ","_")
    task = {**params, "status":status, "timestamp":int(time.time()), "task_id":id}

    added = db['tasks'].insert(task)
    print("task id {} added:{}".format(id,added))
    return task


def delete_tasks():
    db = dataset.connect(DATABASE_URL)
    tasks = db['tasks'].delete()
    return tasks

def get_tasks():
    db = dataset.connect(DATABASE_URL)
    tasks = db['tasks'].all()

    
    return map(_fetch_elements,tasks)

def get_task(id, full=False):
    db = dataset.connect(DATABASE_URL)
    task = db['tasks'].find_one(task_id=id)

    if full:
        task["models"] = _fetch_models(task)
    return task

def run_featurenet(task_id):
    sys.path.append("./")
    from pledge_evolution import PledgeEvolution, base_path
    
    db = dataset.connect(DATABASE_URL)
    task = get_task(task_id)
    base = "{}/{}".format(base_path,task.get("task_name"))
    full_fm_file = task.get("fm")
    output_file = task.get("pdt")
    dt=task.get("dataset")
    nb_base_products=int(task.get("nb_initial_config"))
    training_epochs=int(task.get("nb_training_iterations"))

    products_path = "{}/{}".format(base, dt)
    task["products"] = products_path
    db['tasks'].update(task, ['task_id'])

    PledgeEvolution.run(base, full_fm_file, output_file, dataset=dt, nb_base_products=nb_base_products, training_epochs=training_epochs, evolution_epochs = 0)
    task["status"] = "generation_complete"
    db['tasks'].update(task, ['task_id'])
    print("task id {} generated:{}".format(task["task_id"],products_path))


def run_fm_generation(task_id):
    sys.path.append("./")
    from pledge_evolution import PledgeEvolution, base_path, run_pledge, default_pledge_output
    
    db = dataset.connect(DATABASE_URL)
    task = get_task(task_id)
    base = "{}/{}".format(base_path,task.get("task_name"))
    nb_base_products = (task.get("max_nb_blocks"), task.get("max_nb_cells"), task.get("nb_initial_config"))
    full_fm_file = PledgeEvolution.end2end(base, nb_base_products,  dataset=task.get("dataset"), training_epochs=task.get("nb_training_iterations"))
    max_sampling_time = int(task.get("max_sampling_time",30))

    task["status"] = "fm_complete"
    task["fm"] = full_fm_file
    updated = db['tasks'].update(task, ['task_id'])
    print("task id {} fm built:{}".format(task["task_id"],full_fm_file))


    nb_products = int(task.get("nb_initial_config"))
    output_file = default_pledge_output(base, nb_products)
    pledge_result = run_pledge(full_fm_file, nb_products, output_file, duration=max_sampling_time)

    if pledge_result==0:
        task["status"] = "sampling_complete"
        task["pdt"] = output_file
        updated = db['tasks'].update(task, ['task_id'])
        print("task id {} sampled:{}".format(task["task_id"],output_file))
        
        thread = multiprocessing.Process(target=run_featurenet, args=(task["task_id"],))
        thread.start()
    else:
        task["status"] = "sampling_failed"
        task["pdt"] = output_file
        updated = db['tasks'].update(task, ['task_id'])
        print("task id {} sampled failed".format(task["task_id"]))

def create_app():

    app = Flask(__name__)
    CORS(app)

    @app.route('/sample/', methods=['DELETE'])
    def sample_delete_all():
        tasks = delete_tasks()
        return jsonify(tasks)   

    @app.route('/sample/', methods=['GET'])
    def sample_all():
        tasks = list(get_tasks())
        return jsonify(tasks)   
    

        
    @app.route('/sample/<id>/product/<pId>/<content>', methods=['GET'])
    def model_get(id, pId,content):
        from flask import send_file

        task = get_task(id)
        path = task.get("products")
        file_path = "./{}/".format(path)
        f = None
        mime = None
        if content=="graph":
            f = _get_file_endswith(file_path,"{}.png".format(pId))
            mime = 'image/png'
        else:
            f = _get_file_endswith(file_path,"{}.h5".format(pId))
            mime = 'application/octet-stream'

        if f is not None:
            file_path = "{}{}".format(file_path, f)
            p = os.path.abspath(file_path)
            res = make_response(send_file(p, mime))
            res.headers['Content-Length'] = os.path.getsize(p)
            return res

        return jsonify({})

    @app.route('/sample/<id>', methods=['GET'])
    def sample_get(id):
        data = request.args
        full = bool(data.get("full",False)) if data else False
        
        task = get_task(id,full)
        return jsonify(task)

    @app.route('/sample/', methods=['POST'])
    def sample_post():
        data = request.json
        data = data.get("data")
        task = init_task(data)

        thread = multiprocessing.Process(target=run_fm_generation, args=(task["task_id"],))
        thread.start()

        response = data if data else "00"
        return jsonify(response)

    @app.route('/', defaults={'path': ''})
    @app.route('/<path:path>')
    def catch_all(path):
        return 'You want path: %s' % path
        
    return app


app = create_app()

if __name__ == '__main__':
    host, port = '0.0.0.0', '9999'
    print("running server on {}:{}".format(host,port))
    app.run(host=host, port=port, debug=True)