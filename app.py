

import pandas as pd
from flask import Flask, request, render_template , redirect , url_for
from pipeline.predict_pipeline import DataToPredict


student_performance = Flask(__name__,template_folder="html_templates")

"""
## Route to the landing page '/'
@student_performance.route('/')
def index():
    return render_template('index.html')

"""

# Redirect from the root URL to the /studentperformance URL
@student_performance.route('/')
def redirect_to_studentperformance():
    return redirect(url_for('predict_data'))

## Route to the home page '/home.html'
@student_performance.route('/studentperformance',methods=['GET','POST'])
def predict_data():
    if request.method == 'GET':
        return render_template('home.html')
    else:
        cust_data = DataToPredict( gender = request.form.get('gender'),
                                 race_ethnicity= request.form.get('ethnicity'),
                                 parental_level_of_education = request.form.get('parental_level_of_education'),
                                 test_preparation_course = request.form.get('test_preparation_course'),
                                 reading_score = request.form.get('reading_score'),
                                 writing_score = request.form.get('writing_score'),
                                 ) 

        data_to_predict = cust_data.convert_to_data_frame()

        print(data_to_predict)
        return 

if __name__ == "__main__":
    student_performance.run(host="0.0.0.0",debug=True)