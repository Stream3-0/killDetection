import os
from flask import Flask, request, jsonify 
from clip import Clip 

app = Flask(__name__)


@app.route('/timestamps', methods=['GET'])
def get_timestamps():
    name = request.args.get('name')
    url = request.args.get('url')
    print(name, url)
    clip = Clip.from_url(name, url)
    return jsonify({'timestamps': {'kills': clip.identify_clips()}})


app.run(host="0.0.0.0", debug=True, port=int(os.getenv('PORT')))