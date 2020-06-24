from flask import Flask
import random
import neuron
import json
from flask import request

app = Flask("rater")


@app.route("/test", methods=['GET'])
def test():
    test = []
    for j in range(0, 5):
        test.append(random.uniform(-1, 1))
    return "result %f average %f" % (neuron.calc(test), neuron.average(test))


@app.route("/calc", methods=['POST'])
def calc():
    # list_data = request.json
    list_data = json.loads(request.json)
    list_data = list_data.get("data")
    for i in range(len(list_data), 5):
        list_data.append(float(0))
    return str(neuron.calc(list_data))


if __name__ == "__main__":
    neuron.run()
    app.run(port=1489)
