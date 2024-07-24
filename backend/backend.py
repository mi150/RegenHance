import os
import logging
from flask import Flask, request, jsonify
import yaml
from .server import Server


app = Flask(__name__)
server = None


from munch import *
# 虽然munch 显示的灰色但是一定需要

@app.route("/")
@app.route("/index")
def index():
    # TODO: Add debugging information to the page if needed
    return "Much to do!"


@app.route("/init", methods=["POST"])
def initialize_server():
    args = yaml.load(request.data, Loader=yaml.SafeLoader)
    global server
    if not server:
        logging.basicConfig(
            format="%(name)s -- %(levelname)s -- %(lineno)s -- %(message)s",
            level="INFO")
        server = Server(args, args["nframes"])
        os.makedirs("server_temp", exist_ok=True)
        os.makedirs("server_temp-cropped", exist_ok=True)
        return "New Init"
    else:
        server.reset_state(int(args["nframes"]))
        return "Reset"


@app.route("/low", methods=["POST"])
def low_query():
    file_data=request.files["media"]
    results = server.perform_low_query(file_data)
    return jsonify(results)


@app.route("/ask", methods=["POST"])
def ask_query():
    answer = server.whether_train()
    return jsonify(answer)


@app.route("/high", methods=["POST"])
def high_query():
    file_data = request.files["media"]
    gt_file_data = request.files["gt_media"]
    answer = server.perform_online_training(file_data, gt_file_data)
    return jsonify(answer)
