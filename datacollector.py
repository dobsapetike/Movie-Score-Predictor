import tmdbsimple as tmdb
tmdb.API_KEY = 'XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX'
import urllib
import json, time, os


def loaddata(year):
    page = totalpages = 1
    omdblink = 'http://www.omdbapi.com/?i={}&plot=full&r=json&tomatoes=true'

    data = []

    discover = tmdb.Discover()
    while page <= totalpages:
        try:
            response = discover.movie(year = year, page = page)
            totalpages = int(response['total_pages'])

            movies = response['results']
            for m in movies:
                movie = tmdb.Movies(m['id']).info()
                req = urllib.request.urlopen(omdblink.format(movie['imdb_id']))
                resp = req.read().decode('utf-8')
                addinfo = json.loads(resp)
                movie.update(addinfo)
                if movie['budget'] == 0: continue
                if movie['Response'] == 'False': continue
                if movie['imdbRating'] == 'N/A': continue
                data.append(movie)
                    
            print('progress: {}/{} - {}% ({})'.format(page,totalpages,
                                        round(page/totalpages*100,2),len(data)))
        except Exception as e:
            print('An error has occurred:\n{}'.format(e))
        page += 1
        time.sleep(10)  # because of the request rate limit

    print('completed!')
    return data


data = []
for y in [2012, 2013, 2014]:
    data += loaddata(y)
with open('data.json', 'w') as outfile:
    json.dump(json.dumps(data), outfile)




