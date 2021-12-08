from flask import Flask
from flask_restful import Resource, Api, reqparse

from ranking import run_ranking

parser = reqparse.RequestParser()
parser.add_argument("title", type=str)
parser.add_argument("summary", type=str)
parser.add_argument("k", type=int)

app = Flask(__name__)
api = Api(app)

class SimilarPaper(Resource):
    def get(self):
        args = parser.parse_args()
        rankings = run_ranking(args.title, args.summary, args.k)
        return {"rankings": rankings}

api.add_resource(SimilarPaper, '/')

if __name__ == '__main__':
    app.run(debug=True, port=8000)