from . import *  
from app.irsystem.models.helpers import *
from app.irsystem.models.helpers import NumpyEncoder as NumpyEncoder

project_name = "Used Car Recommendations"
net_id = "Ikra Monjur: im324, Yoon Jae Oh: yo82, Fareeza Hasan: fh244, Destiny Malloy: dam359, David Hu: dsh236"

@irsystem.route('/', methods=['GET'])
def search():
	query = request.args.get('search')
	if not query:
		data = []
		output_message = ''
	else:
		output_message = "Your search: " + query
		data = range(5)
	return render_template('search.html', name=project_name, netid=net_id, output_message=output_message, data=data)



